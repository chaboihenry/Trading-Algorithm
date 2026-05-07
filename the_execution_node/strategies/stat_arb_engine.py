import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from the_research_node.m1_xgboost_trainer import apply_frac_diff, find_optimal_d

from the_utilities.strategy_config import (
    Z_THRESH, AI_THRESH, PT_SKEW, SL_SKEW, TIME_BARRIER, LEVERAGE,
    CUSUM_REGIME_THRESHOLD
)
from the_utilities.paths import (
    MODELS_DIR, CURATED_UNIVERSE_JSON, ACTIVE_MODEL_VERSION,
    META_LABELER_JSON, RAW_MACRO_CSV
)

def _load_meta_labeler():
    # Load the active XGBoost meta-labeler from the version ledger
    try:
        with open(ACTIVE_MODEL_VERSION, "r") as f:
            lines = f.readlines()
        latest = lines[-1].strip()
        model_name = latest.split("Model:")[1].split("|")[0].strip() if "Model:" in latest else latest
        model_path = os.path.join(MODELS_DIR, model_name)
    except Exception:
        model_path = META_LABELER_JSON

    model = xgb.Booster()
    model.load_model(model_path)
    return model


def check_regime_safe(threshold: float = 0.02):
    # CUSUM regime filter on daily SPY returns from the macro CSV
    # Returns False if a structural break was detected on the most recent day
    try:
        macro_df = pd.read_csv(RAW_MACRO_CSV, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"[SHIELD] Could not load {RAW_MACRO_CSV}: {e}. Defaulting to SAFE.")
        return True

    if 'SPY' not in macro_df.columns:
        print("[SHIELD] SPY column missing from macro CSV. Defaulting to SAFE.")
        return True

    returns = macro_df['SPY'].pct_change().dropna()

    if len(returns) < 10:
        return True

    # Run CUSUM on full daily history — only the final state matters
    pos_cusum = 0.0
    neg_cusum = 0.0
    regime_safe = True

    for ret in returns.values:
        pos_cusum = max(0, pos_cusum + ret)
        neg_cusum = min(0, neg_cusum + ret)

        if pos_cusum > threshold or neg_cusum < -threshold:
            regime_safe = False
            pos_cusum = 0.0
            neg_cusum = 0.0
        else:
            regime_safe = True

    return regime_safe


def generate_signals(live_matrix: pd.DataFrame):
    # Evaluate all active spreads and return filtered entry signals
    # Returns active_signals dict and spread_returns DataFrame

    # 1. Load universe
    try:
        with open(CURATED_UNIVERSE_JSON, "r") as f:
            curated_baskets = json.load(f).get("baskets", {})
    except FileNotFoundError:
        print("[CRITICAL] curated_universe.json not found.")
        return {}, pd.DataFrame()

    # 2. CUSUM macro regime filter — blocks ALL entries during structural breaks
    regime_safe = check_regime_safe()
    if not regime_safe:
        print("[SHIELD] CUSUM regime break detected on SPY. Blocking all new entries.")
        return {}, pd.DataFrame()

    # 3. Load meta-labeler once per evaluation cycle
    try:
        meta_labeler = _load_meta_labeler()
    except Exception as e:
        print(f"[WARNING] Meta-labeler load failed: {e}. Signals will be unfiltered.")
        meta_labeler = None

    spread_series = {}
    active_signals = {}

    for spread_name, params in curated_baskets.items():
        weights = params.get('weights', {})
        half_life = params.get('half_life', 1.0)
        allocation = params.get('capital_allocation', 0.0)

        if allocation <= 0:
            continue

        # 4. Verify ALL legs have valid live prices
        missing = [t for t in weights if t not in live_matrix.columns]
        if missing:
            continue

        spread_val = pd.Series(0.0, index=live_matrix.index)
        for ticker, w in weights.items():
            spread_val += live_matrix[ticker] * w

        # Check for stale or NaN prices in any leg
        last_row = live_matrix.iloc[-1]
        if any(pd.isna(last_row.get(t)) or last_row.get(t, 0) <= 0 for t in weights):
            continue

        if spread_val.eq(0).all():
            continue

        spread_series[spread_name] = spread_val

        # 5. Z-score using simple rolling to match trainer and backtester
        window = max(int(half_life * 78), 50)

        if len(spread_val) < window:
            continue

        rolling_mean = spread_val.rolling(window).mean()
        rolling_std = spread_val.rolling(window).std().replace(0, np.nan)
        z_score = (spread_val - rolling_mean) / rolling_std

        current_z = z_score.iloc[-1]
        if pd.isna(current_z):
            continue

        # 6. Generate raw signal from z-score threshold
        target_pos = 0
        if current_z < -Z_THRESH:
            target_pos = 1
        elif current_z > Z_THRESH:
            target_pos = -1

        if target_pos == 0:
            continue

        # 7. Meta-labeler filter — only trade if AI confidence exceeds threshold
        ai_prob = 0.5
        if meta_labeler is not None:
            try:
                # Fractional diff using first-half d to avoid lookahead
                half = len(spread_val) // 2
                opt_d, _ = find_optimal_d(spread_val.iloc[:half])
                spread_fd = apply_frac_diff(spread_val, opt_d)

                features = pd.DataFrame({
                    'frac_diff': spread_fd,
                    'volatility': rolling_std,
                    'signal_strength': z_score,
                }).dropna()

                if not features.empty:
                    dmatrix = xgb.DMatrix(features.iloc[[-1]])
                    ai_prob = float(meta_labeler.predict(dmatrix)[0])
            except Exception as e:
                print(f"  -> [WARNING] Meta-labeler prediction failed for {spread_name}: {e}")
                ai_prob = 0.5

        if ai_prob <= AI_THRESH:
            continue

        # 8. Half-Kelly bet sizing
        kelly_fraction = ai_prob - ((1.0 - ai_prob) / 1.5)
        bet_size = max(0.0, kelly_fraction / 2.0)

        # 9. Compute current volatility for barrier sizing
        vol = spread_val.pct_change().ewm(span=100).std().iloc[-1]
        if pd.isna(vol) or vol <= 0:
            vol = 0.005
        vol_daily = vol * np.sqrt(78)

        active_signals[spread_name] = {
            'johansen_weights': weights,
            'target_position': target_pos,
            'hrp_allocation': allocation,
            'current_z': current_z,
            'ai_confidence': ai_prob,
            'bet_size': bet_size,
            'entry_price': float(spread_val.iloc[-1]),
            'volatility': float(vol_daily),
            'pt_threshold': float(vol_daily * PT_SKEW),
            'sl_threshold': float(vol_daily * SL_SKEW),
            'time_barrier': TIME_BARRIER,
            'entry_bar': 0,
        }

    spread_data = pd.DataFrame(spread_series)
    spread_returns = spread_data.diff().dropna() if not spread_data.empty else pd.DataFrame()

    return active_signals, spread_returns


def check_exits(live_matrix: pd.DataFrame, open_positions: dict):
    # Evaluate open positions for exit conditions
    # Returns dict of spread_name -> exit_reason for positions that should close

    try:
        with open(CURATED_UNIVERSE_JSON, "r") as f:
            curated_baskets = json.load(f).get("baskets", {})
    except FileNotFoundError:
        return {}

    exits = {}

    for spread_name, pos_data in open_positions.items():
        params = curated_baskets.get(spread_name)
        if params is None:
            exits[spread_name] = "basket_removed"
            continue

        weights = params.get('weights', {})
        half_life = params.get('half_life', 1.0)

        # Verify all legs still have live data
        missing = [t for t in weights if t not in live_matrix.columns]
        if missing:
            exits[spread_name] = "missing_legs"
            continue

        # Reconstruct current spread value
        spread_val = pd.Series(0.0, index=live_matrix.index)
        for ticker, w in weights.items():
            spread_val += live_matrix[ticker] * w

        current_price = spread_val.iloc[-1]
        entry_price = pos_data.get('entry_price', 0)
        direction = pos_data.get('target_position', 0)
        bars_held = pos_data.get('bars_held', 0)
        pt_thresh = pos_data.get('pt_threshold', 0.01)
        sl_thresh = pos_data.get('sl_threshold', 0.01)
        time_barrier = pos_data.get('time_barrier', TIME_BARRIER)

        if entry_price == 0 or direction == 0:
            exits[spread_name] = "invalid_position"
            continue

        # 1. Trade return
        trade_return = (current_price / entry_price - 1.0) * direction

        # 2. Profit take
        if trade_return >= pt_thresh:
            exits[spread_name] = "profit_take"
            continue

        # 3. Stop loss
        if trade_return <= -sl_thresh:
            exits[spread_name] = "stop_loss"
            continue

        # 4. Time barrier
        if bars_held >= time_barrier:
            exits[spread_name] = "time_expired"
            continue

        # 5. Mean reversion exit (z crosses zero)
        window = max(int(half_life * 78), 50)
        if len(spread_val) >= window:
            rolling_mean = spread_val.rolling(window).mean()
            rolling_std = spread_val.rolling(window).std().replace(0, np.nan)
            z_score = (spread_val - rolling_mean) / rolling_std
            current_z = z_score.iloc[-1]

            if not pd.isna(current_z):
                if direction == 1 and current_z >= 0:
                    exits[spread_name] = "mean_reversion"
                elif direction == -1 and current_z <= 0:
                    exits[spread_name] = "mean_reversion"

    return exits


if __name__ == "__main__":
    print("\n====== STAT ARB ENGINE DIAGNOSTIC ======")

    np.random.seed(42)
    dummy_tickers = ['AAPL', 'MSFT', 'V', 'MA']
    dates = pd.date_range(end=pd.Timestamp.now(), periods=200, freq='1min')
    prices = np.random.randn(200, len(dummy_tickers)).cumsum(axis=0) + 100
    live_matrix = pd.DataFrame(prices, index=dates, columns=dummy_tickers)

    print("[SYSTEM] Running signal generation on synthetic data...")
    signals, returns = generate_signals(live_matrix)

    if not signals:
        print("[INFO] No signals fired (expected on random data).")
    else:
        for name, data in signals.items():
            action = 'LONG' if data['target_position'] == 1 else 'SHORT'
            print(f"[SIGNAL] {name} | {action} | Z={data['current_z']:.2f} | "
                  f"AI={data['ai_confidence']:.3f} | Bet={data['bet_size']:.3f}")

    print("\n[DONE] Diagnostic complete.")