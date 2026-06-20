"""
Stat-arb backtester — LAYER 3: position + exit state machine (MINIMAL).

Walks the per-bar signals from Layer 2 and turns them into discrete trades:
enter at +/- z_entry (already encoded in raw_signal), exit when the spread
mean-reverts back through z_exit. This minimal variant has NO profit-target,
stop-loss, or time-barrier (later variant) and NO transaction costs (Layer 4).
PnL is accounted in SPREAD units only.

CRITICAL — execution timing (no same-bar lookahead): a signal observed at the
CLOSE of bar t is acted on at bar t+1's spread. Both entries and exits fill at
the NEXT bar, never the bar on which they were detected — same discipline as
the trailing windows in signals.py.
"""

import pandas as pd


def simulate_positions(
    signals: pd.DataFrame,  # output of generate_signals: spread, zscore, raw_signal
    z_exit: float = 0.0,
) -> pd.DataFrame:
    # Work in positional space so "fill at t+1" is unambiguous; carry the
    # DatetimeIndex separately to stamp entry/exit times.
    index = signals.index
    spread = signals["spread"].to_numpy()
    zscore = signals["zscore"].to_numpy()
    raw_signal = signals["raw_signal"].to_numpy()
    n = len(signals)

    trades = []
    position = 0  # 0 flat, +1 long the spread, -1 short the spread
    entry_idx = None
    entry_spread = None

    for t in range(n):
        if position == 0:
            # Detected at bar t's close; can only fill if a t+1 bar exists.
            sig = raw_signal[t]
            if sig != 0 and t + 1 < n:
                position = int(sig)
                entry_idx = t + 1
                entry_spread = spread[entry_idx]
            continue

        # In a position: mean-crossing exit. A long was opened on a very
        # negative z, so it unwinds once z climbs back to z_exit; a short is
        # the mirror image. Detected at bar t, but filled at t+1.
        exit_trigger = (position == 1 and zscore[t] >= z_exit) or (
            position == -1 and zscore[t] <= z_exit
        )
        if exit_trigger and t + 1 < n:
            exit_idx = t + 1
            exit_spread = spread[exit_idx]
            trades.append(
                {
                    "entry_idx": entry_idx,
                    "entry_time": index[entry_idx],
                    "exit_idx": exit_idx,
                    "exit_time": index[exit_idx],
                    "direction": position,
                    "entry_spread": entry_spread,
                    "exit_spread": exit_spread,
                    "bars_held": exit_idx - entry_idx,
                    "exit_reason": "mean_cross",
                    # No costs yet; long profits when spread rises, short when it falls.
                    "spread_pnl": position * (exit_spread - entry_spread),
                }
            )
            position = 0
            entry_idx = None
            entry_spread = None
            # Flat again; re-entry resumes on the NEXT iteration (t+1), so the
            # earliest new fill is t+2 — never the same bar this exit filled.

    # An open position at the tape's end is marked out at the last spread value.
    if position != 0:
        exit_idx = n - 1
        exit_spread = spread[exit_idx]
        trades.append(
            {
                "entry_idx": entry_idx,
                "entry_time": index[entry_idx],
                "exit_idx": exit_idx,
                "exit_time": index[exit_idx],
                "direction": position,
                "entry_spread": entry_spread,
                "exit_spread": exit_spread,
                "bars_held": exit_idx - entry_idx,
                "exit_reason": "end_of_data",
                "spread_pnl": position * (exit_spread - entry_spread),
            }
        )

    columns = [
        "entry_idx", "entry_time", "exit_idx", "exit_time", "direction",
        "entry_spread", "exit_spread", "bars_held", "exit_reason", "spread_pnl",
    ]
    return pd.DataFrame(trades, columns=columns)


if __name__ == "__main__":
    from the_research_node.backtesting.data import (
        load_basket_bars_cached,
        build_spread,
        BASKET,
        WEIGHTS,
    )
    from the_research_node.backtesting.signals import generate_signals

    # Position-layer smoke test only — raw spread-PnL accounting, no costs.
    prices = load_basket_bars_cached(BASKET, "1h")
    spread = build_spread(prices, WEIGHTS)
    signals = generate_signals(spread, half_life_days=0.68)
    trades = simulate_positions(signals)

    print(f"Number of trades: {len(trades)}")
    if not trades.empty:
        print("\nBy direction:")
        print(f"  +1 (long) : {int((trades['direction'] == 1).sum())}")
        print(f"  -1 (short): {int((trades['direction'] == -1).sum())}")

        print("\nBy exit reason:")
        print(trades["exit_reason"].value_counts().to_string())

        print(f"\nBars held — mean: {trades['bars_held'].mean():.2f}, "
              f"median: {trades['bars_held'].median():.1f}")
        print(f"Total spread PnL: {trades['spread_pnl'].sum():.4f}")

        print("\nFirst 5 trades:")
        print(trades.head(5).to_string(index=False))
