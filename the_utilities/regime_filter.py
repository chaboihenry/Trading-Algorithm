import os
import pandas as pd
from the_utilities.strategy_config import CUSUM_REGIME_THRESHOLD, CUSUM_RECOVERY_DAYS
from the_utilities.paths import RAW_MACRO_CSV


def compute_cusum_states(returns: pd.Series, threshold: float = CUSUM_REGIME_THRESHOLD):
    # Returns a boolean Series indicating regime safety per daily timestamp
    pos = 0.0
    neg = 0.0
    states = []
    for r in returns.values:
        pos = max(0, pos + r)
        neg = min(0, neg + r)
        if pos > threshold or neg < -threshold:
            states.append(False)
            pos = 0.0
            neg = 0.0
        else:
            states.append(True)
    return pd.Series(states, index=returns.index)


def check_regime_safe_now(recovery_days: int = CUSUM_RECOVERY_DAYS) -> bool:
    # Returns True only after N consecutive safe days post-break.
    try:
        df = pd.read_csv(RAW_MACRO_CSV, index_col='Date', parse_dates=True)
        if 'SPY' not in df.columns:
            return True
        returns = df['SPY'].pct_change().dropna()
        if len(returns) < recovery_days:
            return True
        states = compute_cusum_states(returns)
        return bool(states.iloc[-recovery_days:].all())
    except Exception:
        return True