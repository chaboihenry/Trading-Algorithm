import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from the_research_node.m1_xgboost_trainer import apply_frac_diff, find_optimal_d


class VectorizedBacktester:
    # Simulates the Execution Node pipeline over historical data in RAM.

    # Monte Carlo optimized parameters
    Z_THRESH = 1.96
    AI_THRESH = 0.55
    PT_SKEW = 1.70
    SL_SKEW = 1.83
    LEVERAGE = 28.5
    WARMUP_BARS = 2340

    def __init__(self, models_dir: str = "the_models", data_dir: str = "the_execution_node/data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.baskets = {}
        self.flat_list = []
        self.meta_labeler = None
        self._load_payloads()

    def _load_payloads(self):
        # Load curated universe and meta-labeler
        universe_path = os.path.join(self.models_dir, "curated_universe.json")
        try:
            with open(universe_path, "r") as f:
                data = json.load(f)
                self.baskets = data.get("baskets", {})
                self.flat_list = data.get("flat_list", [])
            print(f"[SUCCESS] Loaded {len(self.baskets)} baskets from curated_universe.json.")
        except FileNotFoundError:
            print("[CRITICAL] curated_universe.json not found.")
            exit(1)

        xgb_path = os.path.join(self.models_dir, "meta_labeler_v3.json")
        try:
            self.meta_labeler = xgb.Booster()
            self.meta_labeler.load_model(xgb_path)
            print("[SUCCESS] Loaded XGBoost Meta-Labeler (meta_labeler_v3.json).")
        except Exception as e:
            print(f"[WARNING] Could not load XGBoost model. Defaulting to pure Stat-Arb. Error: {e}")
            self.meta_labeler = None

    def _fetch_historical_data(self):
        # Load the 5-year 5-minute parquet matrix
        data_path = os.path.join(self.data_dir, "backtest_5m_5yr.parquet")

        print(f"[SYSTEM] Loading historical Parquet matrix from {data_path}...")
        try:
            df = pd.read_parquet(data_path)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            print(f"[SUCCESS] Historical Matrix Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
            return df.ffill().dropna()
        except FileNotFoundError:
            print(f"[CRITICAL] Historical data vault not found at {data_path}.")
            exit(1)

    def _apply_cusum_regime_shield(self, target_index: pd.DatetimeIndex, threshold: float = 0.02):
        # CUSUM regime filter on daily SPY returns
        csv_path = os.path.join(self.data_dir, "raw_macro_data.csv")

        try:
            macro_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            macro_df.index = pd.to_datetime(macro_df.index, utc=True)
        except Exception as e:
            print(f"[WARNING] Could not load {csv_path}. Defaulting Shield to SAFE. Error: {e}")
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
        aligned_mask = daily_mask.reindex(target_index, method='ffill').fillna(True)
        return aligned_mask

    def run_simulation(self):
        # 1. Ingest Data
        df = self._fetch_historical_data()
        if df.empty:
            print("[CRITICAL] Parquet dataframe is empty.")
            return

        # 2. Run the Shield
        print("[SYSTEM] Calculating CUSUM Market Regime Shield...")
        regime_safe_mask = self._apply_cusum_regime_shield(df.index)

        portfolio_returns = pd.Series(0.0, index=df.index)

        print(f"[SYSTEM] Simulating {len(self.baskets)} HRP-Allocated Strategies...")

        for spread_name, params in self.baskets.items():
            weights = params.get('weights', {})
            half_life = params.get('half_life', 1.0)
            allocation = params.get('capital_allocation', 0.0)

            if allocation <= 0:
                continue

            # Skip incomplete spreads — partial baskets are invalid
            missing = [t for t in weights if t not in df.columns]
            if missing:
                print(f"  -> [SKIP] {spread_name}: missing {missing}")
                continue

            # A. Calculate Synthetic Spread
            spread_val = pd.Series(0.0, index=df.index)
            for ticker, w in weights.items():
                spread_val += df[ticker] * w

            if spread_val.eq(0).all():
                continue

            # B. THE ENGINE: Z-Scores using simple rolling to match training
            # Trainer uses .rolling(), not .ewm()
            window = max(int(half_life * 78), 50)
            rolling_mean = spread_val.rolling(window).mean()
            rolling_std = spread_val.rolling(window).std().replace(0, np.nan)
            z_score = (spread_val - rolling_mean) / rolling_std

            # Use Monte Carlo optimized z-threshold
            base_signals = pd.Series(0, index=df.index)
            base_signals[z_score < -self.Z_THRESH] = 1
            base_signals[z_score > self.Z_THRESH] = -1

            # C. THE BRAIN: XGBoost Meta-Labeling (3-feature v3 model)
            if self.meta_labeler is not None:
                # Use first-half for d search to avoid lookahead bias
                half = len(spread_val) // 2
                opt_d, _ = find_optimal_d(spread_val.iloc[:half])
                spread_fd = apply_frac_diff(spread_val, opt_d)

                features = pd.DataFrame({
                    'frac_diff': spread_fd,
                    'volatility': rolling_std,
                    'signal_strength': z_score,
                }).dropna()

                valid_idx = features.index
                dmatrix = xgb.DMatrix(features)
                win_probs_array = self.meta_labeler.predict(dmatrix)

                # Build aligned Series so Kelly sizing uses correct probs
                win_probs = pd.Series(win_probs_array, index=valid_idx)

                meta_mask = win_probs > self.AI_THRESH

                filtered_signals = pd.Series(0, index=df.index)
                filtered_signals.loc[valid_idx] = base_signals.loc[valid_idx].where(meta_mask, 0)
                base_signals = filtered_signals

            # D. Apply The Shield (Regime Filter)
            final_signals = base_signals.where(regime_safe_mask, 0)

            # E. Event-Driven Execution
            print(f"  -> Simulating {spread_name}...")

            vol = spread_val.pct_change().ewm(span=100).std().fillna(0) * np.sqrt(78)

            in_position = 0
            entry_price = 0.0
            entry_idx = 0
            current_vol = 0.0
            trade_size_multiplier = 0.0

            for i in range(1, len(df)):
                current_price = spread_val.iloc[i]
                prev_price = spread_val.iloc[i - 1]
                signal = final_signals.iloc[i]
                current_z = z_score.iloc[i]
                current_time = df.index[i].time()

                # 1. Entry Logic
                if in_position == 0 and i > self.WARMUP_BARS:
                    is_safe_time = (current_time >= pd.Timestamp("09:45").time()) and \
                                   (current_time <= pd.Timestamp("15:45").time())

                    if signal != 0 and is_safe_time:
                        in_position = signal
                        entry_price = current_price
                        entry_idx = i
                        current_vol = vol.iloc[i] if vol.iloc[i] > 0 else 0.005

                        # Half-Kelly with properly aligned probabilities
                        if self.meta_labeler is not None and df.index[i] in win_probs.index:
                            prob = win_probs.loc[df.index[i]]
                            kelly_fraction = prob - ((1.0 - prob) / 1.5)
                            trade_size_multiplier = max(0.0, kelly_fraction / 2.0)
                        else:
                            trade_size_multiplier = 0.5

                        active_capital = allocation * trade_size_multiplier * self.LEVERAGE
                        portfolio_returns.iloc[i] -= active_capital * 0.0005

                # 2. Manage Open Position
                elif in_position != 0:
                    bars_held = i - entry_idx
                    active_capital = allocation * trade_size_multiplier * self.LEVERAGE

                    if prev_price != 0:
                        tick_ret = (current_price - prev_price) * in_position * active_capital
                        portfolio_returns.iloc[i] += tick_ret

                    if entry_price != 0:
                        trade_return = (current_price / entry_price - 1) * in_position
                    else:
                        trade_return = 0

                    # Monte Carlo optimized barriers
                    hit_pt = trade_return >= (current_vol * self.PT_SKEW)
                    hit_sl = trade_return <= -(current_vol * self.SL_SKEW)
                    hit_time = bars_held >= 120

                    hit_mean_reversion = False
                    if in_position == 1 and current_z >= 0:
                        hit_mean_reversion = True
                    elif in_position == -1 and current_z <= 0:
                        hit_mean_reversion = True

                    if hit_pt or hit_sl or hit_time or hit_mean_reversion:
                        portfolio_returns.iloc[i] -= active_capital * 0.0005
                        in_position = 0
                        entry_price = 0.0
                        entry_idx = 0
                        trade_size_multiplier = 0.0

        # 4. Load SPY baseline
        csv_path = os.path.join(self.data_dir, "raw_macro_data.csv")
        try:
            macro_df = pd.read_csv(csv_path, index_col='Date', parse_dates=True)
            macro_df.index = pd.to_datetime(macro_df.index, utc=True)
            spy_data = macro_df['SPY']
        except Exception as e:
            print(f"[WARNING] Could not load SPY baseline: {e}")
            spy_data = pd.Series(0.0, index=df.index)

        # 5. Generate Tear Sheet
        self._plot_results(portfolio_returns, spy_data)

    def _plot_results(self, portfolio_returns: pd.Series, spy_data: pd.Series):
        print("[SYSTEM] Generating Performance Tear Sheet...")

        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 1])

        # Build aligned Series so Kelly sizing uses correct probs
        active_portfolio = portfolio_returns.iloc[self.WARMUP_BARS:]
        active_spy = spy_data.reindex(active_portfolio.index).ffill()

        cum_returns = active_portfolio.cumsum()
        spy_returns = (active_spy.pct_change().fillna(0).cumsum()) * 100

        ax1 = fig.add_subplot(gs[0])
        ax1.plot(cum_returns.index, cum_returns, color='#00ffcc', linewidth=2, label='Stat-Arb Agent')
        ax1.plot(spy_returns.index, spy_returns, color='#555555', linewidth=1, linestyle='--', label='SPY Baseline (%)')
        ax1.set_title('5-Year Out-of-Sample Performance Estimate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.grid(color='#222222', linestyle='--')
        ax1.legend(loc='upper left')

        rolling_max = cum_returns.cummax()
        drawdown = cum_returns - rolling_max

        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.fill_between(drawdown.index, drawdown, 0, color='#ff3366', alpha=0.5)
        ax2.set_ylabel('Drawdown ($)')
        ax2.set_title('Underwater Curve', fontsize=12)
        ax2.grid(color='#222222', linestyle='--')

        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.bar(active_portfolio.index, active_portfolio, color='#ffcc00', alpha=0.7)
        ax3.set_ylabel('Tick Returns')
        ax3.set_title('5-Minute Interval Returns Distribution', fontsize=12)
        ax3.grid(color='#222222', linestyle='--')

        plt.tight_layout()

        output_image = "backtest_tearsheet_5yr.png"
        plt.savefig(output_image, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
        print(f"\n[SUCCESS] Tear Sheet saved as '{output_image}'")

        # Console Metrics
        total_pl = cum_returns.iloc[-1]
        max_dd = drawdown.min()
        romd = abs(total_pl / max_dd) if max_dd < 0 else 0
        active_intervals = len(active_portfolio[active_portfolio != 0])

        print("\n====== ESTIMATED HISTORICAL PERFORMANCE ======")
        print(f"Total Cumulative P&L: ${total_pl:.2f}")
        print(f"Max Drawdown:         ${max_dd:.2f}")
        print(f"RoMD:                 {romd:.2f}")
        print(f"Trade Density:        {active_intervals} active 5-min intervals")
        print(f"Parameters:           z={self.Z_THRESH} ai={self.AI_THRESH} "
              f"pt={self.PT_SKEW} sl={self.SL_SKEW} lev={self.LEVERAGE}")
        print("==============================================")


if __name__ == "__main__":
    backtester = VectorizedBacktester()
    backtester.run_simulation()