"""
Stat-arb backtester — LAYER 2: point-in-time signal generation.

Turns the reconstructed spread into a trailing z-score and a raw directional
signal. This layer ONLY emits signals; the hold/exit state machine, position
simulation, costs, and metrics belong to later layers.

CRITICAL — no lookahead: every value at bar t uses ONLY bars <= t. The
rolling statistics are strictly trailing windows (never the full series), so
the signal at t could have been computed live at t with no future knowledge.
"""

import pandas as pd

# Bars per 6.5-hour trading session, by bar frequency. "1h" -> 8 = the
# measured RTH hourly bars per 6.5h session (09:30-16:00 ET, inclusive of the
# partial open hour).
BARS_PER_DAY = {"1h": 8, "30min": 13, "2h": 4, "1d": 1}


def generate_signals(
    spread: pd.Series,
    half_life_days: float,
    freq: str = "1h",
    window_hl_mult: float = 3.0,
    z_entry: float = 2.0,
    z_exit: float = 0.0,
) -> pd.DataFrame:
    # Adaptability principle: the lookback is a function of the basket's OWN
    # measured half-life, not a tuned absolute — faster-reverting baskets get
    # shorter windows automatically. Floor at 5 bars so std() is meaningful.
    bars_per_day = BARS_PER_DAY[freq]
    window = max(int(window_hl_mult * half_life_days * bars_per_day), 5)

    # Trailing stats only. min_periods=window means the leading bars before a
    # full window exists stay NaN rather than leaking a partial-window estimate.
    roll_mean = spread.rolling(window, min_periods=window).mean()
    roll_std = spread.rolling(window, min_periods=window).std()
    zscore = (spread - roll_mean) / roll_std

    # Raw entry signal from the z-score (Layer 3 owns the hold/exit logic):
    #   +1 long the spread when it is cheap (z <= -z_entry)
    #   -1 short the spread when it is rich (z >= z_entry)
    #    0 otherwise
    raw_signal = pd.Series(0, index=spread.index, dtype=int)
    raw_signal[zscore <= -z_entry] = 1
    raw_signal[zscore >= z_entry] = -1

    df = pd.DataFrame(
        {"spread": spread, "zscore": zscore, "raw_signal": raw_signal},
        index=spread.index,
    )

    # Stash resolved window + params so downstream layers stay traceable.
    df.attrs["window"] = window
    df.attrs["half_life_days"] = half_life_days
    df.attrs["freq"] = freq
    df.attrs["window_hl_mult"] = window_hl_mult
    df.attrs["z_entry"] = z_entry
    df.attrs["z_exit"] = z_exit
    return df


if __name__ == "__main__":
    from the_research_node.backtesting.data import (
        load_basket_bars_cached,
        build_spread,
        BASKET,
        WEIGHTS,
    )

    # Signal-layer smoke test only — nothing is traded here.
    prices = load_basket_bars_cached(BASKET, "1h")
    spread = build_spread(prices, WEIGHTS)

    # half_life_days=0.68 is this basket's measured OU half-life.
    signals = generate_signals(spread, half_life_days=0.68)

    print(f"Resolved window: {signals.attrs['window']} bars")
    print("\nZ-score describe():")
    print(signals["zscore"].describe())

    counts = signals["raw_signal"].value_counts()
    print("\nRaw signal counts:")
    print(f"  +1 (long) : {int(counts.get(1, 0))}")
    print(f"  -1 (short): {int(counts.get(-1, 0))}")
    print(f"   0 (flat) : {int(counts.get(0, 0))}")

    print("\nFirst 5 non-NaN rows:")
    print(signals.dropna().head(5))
