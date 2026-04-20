import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from the_research_node.m1_xgboost_trainer import apply_frac_diff, find_optimal_d


class VectorizedBacktester:
    # Simulates the Execution Node pipeline over historical data.
    # Lifecycle-aware simulation using structural profiler ledger.

    # Monte Carlo optimized parameters (must match stat_arb_engine)
    Z_THRESH = 2.39
    AI_THRESH = 0.56
    PT_SKEW = 1.90
    SL_SKEW = 1.75
    TIME_BARRIER = 120

    # Reg-T Pattern Day Trader margin — 2x for longs, 2x for shorts combined
    LEVERAGE = 2.0
    WARMUP_BARS = 2340

    # Starting capital — matches the Alpaca paper account
    STARTING_EQUITY = 100_000.0

    # Per-leg slippage on market orders (one-way cost)
    SLIPPAGE_BPS = 7.0

    # cooldown tracking — 30 min / 5 min per bar = 6 bars
    COOLDOWN_BARS = 6

    # Non-shortable tickers (mirrors Alpaca's restrictions — SO is the known one)
    NON_SHORTABLE = {'SO'}

    # Annualized short borrow cost
    SHORT_BORROW_APR = 0.01
    BARS_PER_YEAR = 252 * 78

    # equal-weight allocation across baskets active in any given window
    EQUAL_WEIGHT_ALLOCATION = True

    def __init__(self, models_dir: str = "the_models", data_dir: str = "the_execution_node/data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.ledger = {}
        self.meta_labeler = None
        self._load_payloads()
        # shared capital state
        self.cash = self.STARTING_EQUITY
        self.held_tickers = {}
        # cooldown state (keyed by underlying spread_name, not window key)
        self.cooldown_until = {}

    def _load_payloads(self):
        # load lifecycle ledger instead of curated_universe
        ledger_path = os.path.join(self.models_dir, "structural_lifecycle_5yr.json")
        try:
            with open(ledger_path, "r") as f:
                self.ledger = json.load(f)
            print(f"[SUCCESS] Loaded {len(self.ledger)} historical baskets from structural_lifecycle_5yr.json.")
        except FileNotFoundError:
            print("[CRITICAL] structural_lifecycle_5yr.json not found — run m1_structural_profiler first.")
            raise

        xgb_path = os.path.join(self.models_dir, "meta_labeler_v3.json")
        try:
            self.meta_labeler = xgb.Booster()
            self.meta_labeler.load_model(xgb_path)
            print("[SUCCESS] Loaded XGBoost Meta-Labeler.")
        except Exception as e:
            print(f"[WARNING] Could not load XGBoost model. Error: {e}")
            self.meta_labeler = None

    def _fetch_historical_data(self):
        data_path = os.path.join(self.data_dir, "backtest_5m_5yr.parquet")
        print(f"[SYSTEM] Loading historical matrix from {data_path}...")
        try:
            df = pd.read_parquet(data_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            # Normalize to UTC so timestamps match the profiler's lifecycle windows
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
            print(f"[SUCCESS] Historical matrix: {df.shape[0]} rows x {df.shape[1]} cols")
            return df.ffill().dropna()
        except FileNotFoundError:
            print(f"[CRITICAL] Historical data not found at {data_path}.")
            raise

    def _apply_cusum_regime_shield(self, target_index, threshold: float = 0.02):
        csv_path = os.path.join(self.data_dir, "raw_macro_data.csv")
        try:
            macro_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            macro_df.index = pd.to_datetime(macro_df.index, utc=True)
        except Exception as e:
            print(f"[WARNING] Could not load macro CSV: {e}. Shield defaulting SAFE.")
            return pd.Series(True, index=target_index)

        spy_daily = macro_df['SPY']
        returns = spy_daily.pct_change().dropna()

        pos_cusum = np.zeros_like(returns)
        neg_cusum = np.zeros_like(returns)
        regime_safe = np.ones_like(returns, dtype=bool)

        for i in range(1, len(returns)):
            pos_cusum[i] = max(0, pos_cusum[i - 1] + returns.iloc[i])
            neg_cusum[i] = min(0, neg_cusum[i - 1] + returns.iloc[i])
            if pos_cusum[i] > threshold or neg_cusum[i] < -threshold:
                regime_safe[i] = False
                pos_cusum[i] = 0
                neg_cusum[i] = 0

        daily_mask = pd.Series(np.insert(regime_safe, 0, True), index=spy_daily.index)
        return daily_mask.reindex(target_index, method='ffill').fillna(True)

    def _size_legs(self, signal_data: dict, current_prices: pd.Series, current_equity: float):
        # Mirrors order_router.execute_spread sizing
        target_pos = signal_data['target_position']
        allocation = signal_data['hrp_allocation']
        bet_size = signal_data['bet_size']
        weights = signal_data['johansen_weights']

        spread_capital = current_equity * allocation * bet_size * self.LEVERAGE

        abs_weight_sum = sum(abs(w) for w in weights.values())
        if abs_weight_sum == 0:
            return {}, 0.0

        leg_shares = {}
        total_notional = 0.0
        for ticker, weight in weights.items():
            price = current_prices.get(ticker, 0.0)
            if pd.isna(price) or price <= 0:
                continue

            leg_capital = spread_capital * (abs(weight) / abs_weight_sum)
            trade_direction = target_pos * weight
            side = 1 if trade_direction > 0 else -1

            qty = math.floor(leg_capital / price)
            if qty <= 0:
                return {}, 0.0

            leg_shares[ticker] = side * qty
            total_notional += abs(qty) * price

        return leg_shares, total_notional

    def _mark_to_market(self, positions: dict, current_prices: pd.Series):
        total_unrealized = 0.0
        for spread_name, legs in positions.items():
            for ticker, (shares, entry_price) in legs.items():
                current_price = current_prices.get(ticker, entry_price)
                if pd.isna(current_price):
                    current_price = entry_price
                total_unrealized += shares * (current_price - entry_price)
        return total_unrealized

    def _close_spread(self, key: str, positions: dict, current_prices: pd.Series, current_bar: int):
        # key is the window-indexed key like "V_MA_Spread#2"
        # Extract the underlying spread_name for the cooldown tracker
        underlying_spread = key.split('#')[0]

        if key not in positions:
            return 0.0

        realized = 0.0
        slippage_cost = 0.0
        for ticker, (shares, entry_price) in positions[key].items():
            exit_price = current_prices.get(ticker, entry_price)
            if pd.isna(exit_price):
                exit_price = entry_price

            realized += shares * (exit_price - entry_price)
            slippage_cost += abs(shares) * exit_price * (self.SLIPPAGE_BPS / 10000.0)
            self.held_tickers.pop(ticker, None)

        del positions[key]
        self.cash += realized - slippage_cost
        # Cooldown applies to ALL windows of this spread, not just this one
        self.cooldown_until[underlying_spread] = current_bar + self.COOLDOWN_BARS
        return realized - slippage_cost

    def _accrue_short_borrow(self, positions: dict, current_prices: pd.Series):
        borrow_cost = 0.0
        per_bar_rate = self.SHORT_BORROW_APR / self.BARS_PER_YEAR
        for legs in positions.values():
            for ticker, (shares, entry_px) in legs.items():
                if shares < 0:
                    px = current_prices.get(ticker, entry_px)
                    if not pd.isna(px):
                        borrow_cost += abs(shares) * px * per_bar_rate
        self.cash -= borrow_cost
        return borrow_cost

    def run_simulation(self):
        df = self._fetch_historical_data()
        if df.empty:
            print("[CRITICAL] Empty dataframe.")
            return

        print("[SYSTEM] Calculating CUSUM regime shield...")
        regime_safe_mask = self._apply_cusum_regime_shield(df.index)

        spread_metadata = self._precompute_spreads(df)
        # precompute how many lifecycle windows are active at each bar
        active_count_series = pd.Series(0, index=df.index, dtype=int)
        for meta in spread_metadata.values():
            mask = (df.index >= meta['window_start']) & (df.index <= meta['window_end'])
            active_count_series.loc[mask] += 1
        active_count_series = active_count_series.clip(lower=1)
        print(f"[SYSTEM] {len(spread_metadata)} lifecycle windows pre-computed.")

        positions = {}  # key -> {ticker: (shares, entry_price)}
        realized_pnl = pd.Series(0.0, index=df.index)

        print("[SYSTEM] Running event-driven simulation...")

        for i in range(self.WARMUP_BARS, len(df)):
            current_prices = df.iloc[i]
            current_time = df.index[i].time()
            is_safe_time = (pd.Timestamp("09:45").time() <= current_time <= pd.Timestamp("15:45").time())
            is_regime_safe = regime_safe_mask.iloc[i] if i < len(regime_safe_mask) else True

            # accrue short borrow costs each bar
            borrow = self._accrue_short_borrow(positions, current_prices)
            realized_pnl.iloc[i] -= borrow

            # --- EXIT PASS ---
            for key in list(positions.keys()):
                meta = spread_metadata.get(key)
                if meta is None:
                    continue

                entry_bar = meta['entry_bars'].get(key, i)
                bars_held = i - entry_bar
                direction = meta['direction'].get(key, 0)

                # Compute spread return from ticker-level P&L (share-based)
                spread_pnl = 0.0
                spread_cost = 0.0
                for ticker, (shares, entry_px) in positions[key].items():
                    px_now = current_prices.get(ticker, entry_px)
                    if not pd.isna(px_now):
                        spread_pnl += shares * (px_now - entry_px)
                        spread_cost += abs(shares) * entry_px

                trade_return = spread_pnl / spread_cost if spread_cost > 0 else 0.0

                # datetime-based lookups
                current_vol = meta['vol'].get(df.index[i], 0.005)
                current_z = meta['z_score'].get(df.index[i], 0.0)

                hit_pt = trade_return >= (current_vol * self.PT_SKEW)
                hit_sl = trade_return <= -(current_vol * self.SL_SKEW)
                hit_time = bars_held >= self.TIME_BARRIER
                hit_mr = (direction == 1 and current_z >= 0) or (direction == -1 and current_z <= 0)

                # force-exit when lifecycle window ends
                window_end = meta.get('window_end')
                hit_window_expiry = (window_end is not None and df.index[i] >= window_end)

                if hit_pt or hit_sl or hit_time or hit_mr or hit_window_expiry:
                    realized_pnl.iloc[i] += self._close_spread(key, positions, current_prices, i)

            # EOD force-liquidation at 15:50
            if current_time >= pd.Timestamp("15:50").time():
                for key in list(positions.keys()):
                    realized_pnl.iloc[i] += self._close_spread(key, positions, current_prices, i)

            # --- ENTRY PASS ---
            if not is_safe_time or not is_regime_safe:
                continue

            for key, meta in spread_metadata.items():
                if key in positions:
                    continue

                # cooldown is per underlying spread, not per window
                underlying_spread = meta['spread_name']
                if i < self.cooldown_until.get(underlying_spread, 0):
                    continue

                # only fire entries within the active lifecycle window
                if df.index[i] < meta['window_start'] or df.index[i] > meta['window_end']:
                    continue

                # datetime-based lookups
                signal = meta['signals'].get(df.index[i], 0)
                if signal == 0:
                    continue

                # count active windows at this bar for equal-weight allocation
                dynamic_alloc = 1.0 / active_count_series.iloc[i]

                signal_data = {
                    'target_position': int(signal),
                    'hrp_allocation': dynamic_alloc,
                    'bet_size': meta['bet_sizes'].get(df.index[i], 0.5),
                    'johansen_weights': meta['weights'],
                }

                unrealized = self._mark_to_market(positions, current_prices)
                current_equity = self.cash + unrealized

                leg_shares, total_notional = self._size_legs(signal_data, current_prices, current_equity)
                if not leg_shares:
                    continue

                # shortability pre-check
                if any(ticker in self.NON_SHORTABLE and shares < 0 for ticker, shares in leg_shares.items()):
                    continue

                # SHIELD: block if notional exceeds available buying power
                deployed_notional = sum(
                    abs(shares) * current_prices.get(ticker, entry_px)
                    for legs in positions.values()
                    for ticker, (shares, entry_px) in legs.items()
                )
                available_bp = current_equity * self.LEVERAGE - deployed_notional
                if total_notional > available_bp:
                    continue

                # don't net positions across spreads
                if any(ticker in self.held_tickers for ticker in leg_shares):
                    continue

                # Open the position
                positions[key] = {
                    ticker: (shares, current_prices[ticker])
                    for ticker, shares in leg_shares.items()
                }
                for ticker, shares in leg_shares.items():
                    self.held_tickers[ticker] = shares

                meta['entry_bars'][key] = i
                meta['direction'][key] = int(signal)

                # Entry slippage
                entry_cost = sum(
                    abs(shares) * current_prices[ticker] * (self.SLIPPAGE_BPS / 10000.0)
                    for ticker, shares in leg_shares.items()
                )
                self.cash -= entry_cost
                realized_pnl.iloc[i] -= entry_cost

        print("[SYSTEM] Building equity curve...")
        equity_curve = self.STARTING_EQUITY + realized_pnl.cumsum()

        csv_path = os.path.join(self.data_dir, "raw_macro_data.csv")
        try:
            macro_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            macro_df.index = pd.to_datetime(macro_df.index, utc=True)
            spy_data = macro_df['SPY']
        except Exception:
            spy_data = pd.Series(0.0, index=df.index)

        self._plot_results(equity_curve, spy_data)

    def run_simulation_headless(self):
        # Same as run_simulation but skips plotting and returns metrics only
        # Used for Monte Carlo optimization where hundreds of configs are run
        df = self._fetch_historical_data()
        if df.empty:
            return 0.0, 0.0, 0

        regime_safe_mask = self._apply_cusum_regime_shield(df.index)
        spread_metadata = self._precompute_spreads(df)

        active_count_series = pd.Series(0, index=df.index, dtype=int)
        for meta in spread_metadata.values():
            mask = (df.index >= meta['window_start']) & (df.index <= meta['window_end'])
            active_count_series.loc[mask] += 1
        active_count_series = active_count_series.clip(lower=1)

        positions = {}
        realized_pnl = pd.Series(0.0, index=df.index)
        trade_count = 0

        # Reset state in case the instance was reused
        self.cash = self.STARTING_EQUITY
        self.held_tickers = {}
        self.cooldown_until = {}

        for i in range(self.WARMUP_BARS, len(df)):
            current_prices = df.iloc[i]
            current_time = df.index[i].time()
            is_safe_time = (pd.Timestamp("09:45").time() <= current_time <= pd.Timestamp("15:45").time())
            is_regime_safe = regime_safe_mask.iloc[i] if i < len(regime_safe_mask) else True

            borrow = self._accrue_short_borrow(positions, current_prices)
            realized_pnl.iloc[i] -= borrow

            # EXIT PASS
            for key in list(positions.keys()):
                meta = spread_metadata.get(key)
                if meta is None:
                    continue
                entry_bar = meta['entry_bars'].get(key, i)
                bars_held = i - entry_bar
                direction = meta['direction'].get(key, 0)
                spread_pnl = 0.0
                spread_cost = 0.0
                for ticker, (shares, entry_px) in positions[key].items():
                    px_now = current_prices.get(ticker, entry_px)
                    if not pd.isna(px_now):
                        spread_pnl += shares * (px_now - entry_px)
                        spread_cost += abs(shares) * entry_px
                trade_return = spread_pnl / spread_cost if spread_cost > 0 else 0.0
                current_vol = meta['vol'].get(df.index[i], 0.005)
                current_z = meta['z_score'].get(df.index[i], 0.0)
                hit_pt = trade_return >= (current_vol * self.PT_SKEW)
                hit_sl = trade_return <= -(current_vol * self.SL_SKEW)
                hit_time = bars_held >= self.TIME_BARRIER
                hit_mr = (direction == 1 and current_z >= 0) or (direction == -1 and current_z <= 0)
                window_end = meta.get('window_end')
                hit_window_expiry = (window_end is not None and df.index[i] >= window_end)
                if hit_pt or hit_sl or hit_time or hit_mr or hit_window_expiry:
                    realized_pnl.iloc[i] += self._close_spread(key, positions, current_prices, i)

            if current_time >= pd.Timestamp("15:50").time():
                for key in list(positions.keys()):
                    realized_pnl.iloc[i] += self._close_spread(key, positions, current_prices, i)

            if not is_safe_time or not is_regime_safe:
                continue

            for key, meta in spread_metadata.items():
                if key in positions:
                    continue
                underlying_spread = meta['spread_name']
                if i < self.cooldown_until.get(underlying_spread, 0):
                    continue
                if df.index[i] < meta['window_start'] or df.index[i] > meta['window_end']:
                    continue
                signal = meta['signals'].get(df.index[i], 0)
                if signal == 0:
                    continue
                dynamic_alloc = 1.0 / active_count_series.iloc[i]
                signal_data = {
                    'target_position': int(signal),
                    'hrp_allocation': dynamic_alloc,
                    'bet_size': meta['bet_sizes'].get(df.index[i], 0.5),
                    'johansen_weights': meta['weights'],
                }
                unrealized = self._mark_to_market(positions, current_prices)
                current_equity = self.cash + unrealized
                leg_shares, total_notional = self._size_legs(signal_data, current_prices, current_equity)
                if not leg_shares:
                    continue
                if any(ticker in self.NON_SHORTABLE and shares < 0 for ticker, shares in leg_shares.items()):
                    continue
                deployed_notional = sum(
                    abs(shares) * current_prices.get(ticker, entry_px)
                    for legs in positions.values()
                    for ticker, (shares, entry_px) in legs.items()
                )
                available_bp = current_equity * self.LEVERAGE - deployed_notional
                if total_notional > available_bp:
                    continue
                if any(ticker in self.held_tickers for ticker in leg_shares):
                    continue
                positions[key] = {
                    ticker: (shares, current_prices[ticker])
                    for ticker, shares in leg_shares.items()
                }
                for ticker, shares in leg_shares.items():
                    self.held_tickers[ticker] = shares
                meta['entry_bars'][key] = i
                meta['direction'][key] = int(signal)
                entry_cost = sum(
                    abs(shares) * current_prices[ticker] * (self.SLIPPAGE_BPS / 10000.0)
                    for ticker, shares in leg_shares.items()
                )
                self.cash -= entry_cost
                realized_pnl.iloc[i] -= entry_cost
                trade_count += 1

        equity_curve = self.STARTING_EQUITY + realized_pnl.cumsum()
        total_return_pct = (equity_curve.iloc[-1] / self.STARTING_EQUITY - 1.0) * 100
        rolling_max = equity_curve.cummax()
        drawdown_pct = (equity_curve - rolling_max) / rolling_max * 100
        max_dd_pct = drawdown_pct.min()

        return total_return_pct, max_dd_pct, trade_count

    def _precompute_spreads(self, df: pd.DataFrame):
        # Precompute per-lifecycle-window, keyed by "spread_name#window_idx"
        spread_metadata = {}
        
        for spread_name, basket_data in self.ledger.items():
            tickers = basket_data['tickers']
            lifecycle = basket_data['lifecycle']

            missing = [t for t in tickers if t not in df.columns]
            if missing:
                print(f"  -> [SKIP] {spread_name}: missing {missing} from parquet.")
                continue

            for window_idx, window in enumerate(lifecycle):
                key = f"{spread_name}#{window_idx}"
                window_start = pd.Timestamp(window['start'])
                window_end = pd.Timestamp(window['end'])
                weights = window['weights']
                half_life = window['half_life']

                buffer_start = window_start - pd.Timedelta(days=30)
                window_df = df.loc[buffer_start:window_end]
                if window_df.empty:
                    continue

                spread_val = pd.Series(0.0, index=window_df.index)
                for ticker, w in weights.items():
                    spread_val += window_df[ticker] * w
                if spread_val.eq(0).all():
                    continue

                z_window = max(int(half_life * 78), 50)
                rolling_mean = spread_val.rolling(z_window).mean()
                rolling_std = spread_val.rolling(z_window).std().replace(0, np.nan)
                z_score = (spread_val - rolling_mean) / rolling_std

                raw_signals = pd.Series(0, index=window_df.index)
                raw_signals[z_score < -self.Z_THRESH] = 1
                raw_signals[z_score > self.Z_THRESH] = -1

                bet_sizes = pd.Series(0.5, index=window_df.index)
                if self.meta_labeler is not None:
                    half = len(spread_val) // 2
                    if half >= 100:
                        opt_d, _ = find_optimal_d(spread_val.iloc[:half])
                        spread_fd = apply_frac_diff(spread_val, opt_d)

                        features = pd.DataFrame({
                            'frac_diff': spread_fd,
                            'volatility': rolling_std,
                            'signal_strength': z_score,
                        }).dropna()

                        if not features.empty:
                            probs = self.meta_labeler.predict(xgb.DMatrix(features))
                            win_probs = pd.Series(probs, index=features.index)
                            meta_mask = win_probs > self.AI_THRESH
                            raw_signals.loc[features.index] = raw_signals.loc[features.index].where(meta_mask, 0)
                            kelly = win_probs - ((1.0 - win_probs) / 1.5)
                            bet_sizes.loc[features.index] = (kelly / 2.0).clip(lower=0.0)

                vol = spread_val.pct_change().ewm(span=100).std().fillna(0) * np.sqrt(78)

                # Only allow signals within the lifecycle window (not buffer)
                raw_signals.loc[raw_signals.index < window_start] = 0
                raw_signals.loc[raw_signals.index > window_end] = 0

                spread_metadata[key] = {
                    'spread_name': spread_name,
                    'weights': weights,
                    'allocation': 1.0,  # per-bar scaling happens in entry pass
                    'signals': raw_signals,
                    'z_score': z_score,
                    'vol': vol,
                    'bet_sizes': bet_sizes,
                    'window_start': window_start,
                    'window_end': window_end,
                    'entry_bars': {},
                    'direction': {},
                }

        return spread_metadata

    def _plot_results(self, equity_curve: pd.Series, spy_data: pd.Series):
        # Final deploy chart: agent vs SPY baseline with summary stats
        print("[SYSTEM] Generating tear sheet...")
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 1, height_ratios=[2, 1])

        active = equity_curve.iloc[self.WARMUP_BARS:]
        active_spy = spy_data.reindex(active.index).ffill().bfill()

        # Normalize both to % return from their first value
        agent_pct = (active / self.STARTING_EQUITY - 1.0) * 100
        spy_pct = (active_spy / active_spy.iloc[0] - 1.0) * 100 if len(active_spy) else pd.Series()

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(agent_pct.index, agent_pct, color='#00ffcc', linewidth=2.5,
                 label=f'Stat-Arb Agent ({self.LEVERAGE}x leverage)')
        ax1.plot(spy_pct.index, spy_pct, color='#888', linewidth=1.5,
                 linestyle='--', label='SPY Buy & Hold')
        ax1.axhline(y=0, color='#444', linewidth=0.8, linestyle=':')
        ax1.set_title('5-Year Backtest: Stat-Arb Agent vs SPY Baseline',
                      fontsize=15, fontweight='bold', pad=15)
        ax1.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax1.grid(color='#222', linestyle='--', alpha=0.5)
        ax1.legend(loc='upper left', fontsize=11, framealpha=0.8)

        # Annotate final values
        agent_final = agent_pct.iloc[-1]
        spy_final = spy_pct.iloc[-1] if len(spy_pct) else 0.0
        ax1.annotate(f'Agent: {agent_final:.2f}%',
                     xy=(agent_pct.index[-1], agent_final),
                     xytext=(-100, 10), textcoords='offset points',
                     fontsize=11, color='#00ffcc', fontweight='bold')
        ax1.annotate(f'SPY: {spy_final:.2f}%',
                     xy=(spy_pct.index[-1], spy_final),
                     xytext=(-90, -20), textcoords='offset points',
                     fontsize=11, color='#aaa')

        # Drawdown panel
        rolling_max = equity_curve.cummax()
        drawdown_pct = (equity_curve - rolling_max) / rolling_max * 100
        active_dd = drawdown_pct.iloc[self.WARMUP_BARS:]

        # SPY drawdown for comparison
        spy_rolling_max = active_spy.cummax()
        spy_dd = (active_spy - spy_rolling_max) / spy_rolling_max * 100

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(active_dd.index, active_dd, 0, color='#00ffcc',
                         alpha=0.4, label=f'Agent DD (max {active_dd.min():.2f}%)')
        ax2.fill_between(spy_dd.index, spy_dd, 0, color='#888',
                         alpha=0.3, label=f'SPY DD (max {spy_dd.min():.2f}%)')
        ax2.axhline(y=0, color='#444', linewidth=0.8)
        ax2.set_ylabel('Drawdown (%)', fontsize=11)
        ax2.set_title('Underwater Curves — Agent vs SPY', fontsize=12)
        ax2.grid(color='#222', linestyle='--', alpha=0.5)
        ax2.legend(loc='lower left', fontsize=10, framealpha=0.8)

        plt.tight_layout()
        plt.savefig("backtest_tearsheet_final.png", dpi=300,
                    bbox_inches='tight', facecolor=fig.get_facecolor())

        total_return = agent_final
        max_dd = active_dd.min()
        romd = abs(total_return / max_dd) if max_dd < 0 else 0.0
        spy_return = spy_final
        spy_max_dd = spy_dd.min()

        print("\n" + "=" * 62)
        print("  FINAL BACKTEST RESULTS — OPTIMIZED PARAMETERS")
        print("=" * 62)
        print(f"  Period:              {active.index[0].date()} to {active.index[-1].date()}")
        print(f"  Starting Equity:     ${self.STARTING_EQUITY:,.2f}")
        print(f"  Ending Equity:       ${equity_curve.iloc[-1]:,.2f}")
        print()
        print(f"  Agent Total Return:  {total_return:.2f}%")
        print(f"  SPY Total Return:    {spy_return:.2f}%")
        print()
        print(f"  Agent Max Drawdown:  {max_dd:.2f}%")
        print(f"  SPY Max Drawdown:    {spy_max_dd:.2f}%")
        print()
        print(f"  Agent RoMD:          {romd:.2f}")
        print()
        print(f"  Parameters: Z={self.Z_THRESH} AI={self.AI_THRESH} "
              f"PT={self.PT_SKEW} SL={self.SL_SKEW} Lev={self.LEVERAGE}x")
        print("=" * 62)


if __name__ == "__main__":
    VectorizedBacktester().run_simulation()

