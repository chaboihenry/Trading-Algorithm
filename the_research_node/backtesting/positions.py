"""
Stat-arb backtester — LAYER 3: position + exit state machine (MINIMAL).

Walks the per-bar signals from Layer 2 and turns them into discrete trades:
enter at +/- z_entry (already encoded in raw_signal), exit when the spread
mean-reverts back through z_exit. This minimal variant has NO profit-target,
stop-loss, or time-barrier (later variant) and NO transaction costs (Layer 4).
PnL is accounted in SPREAD units only.

Execution timing (no same-bar lookahead): a signal observed at the
CLOSE of bar t is acted on at bar t+1's spread. Both entries and exits fill at
the next bar, never the bar on which they were detected — same discipline as
the trailing windows in signals.py.
"""

import pandas as pd


def simulate_positions(
    signals: pd.DataFrame,  # output of generate_signals: spread, zscore, raw_signal
    z_exit: float = 0.0,
    half_life_days: float | None = None,  # required if time barrier used
    freq: str = "1h",
    time_hl_mult: float | None = None,  # None = no time barrier (minimal)
) -> pd.DataFrame:
    # Optional time barrier. For a mean-reverting spread the only sound risk
    # control is elapsed time, not adverse price: an adverse move strengthens
    # the reversion thesis, so only failure-to-revert (too many bars held)
    # signals a broken relationship. Derived from the basket's own half-life.
    time_barrier_bars = None
    if time_hl_mult is not None:
        if half_life_days is None:
            raise ValueError("half_life_days is required when time_hl_mult is set")
        from the_research_node.backtesting.signals import BARS_PER_DAY
        time_barrier_bars = max(int(time_hl_mult * half_life_days * BARS_PER_DAY[freq]), 1)

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

        # In a position, evaluate exits in PRIORITY ORDER (detected at bar t,
        # filled at t+1 in every case):
        #   1. time barrier — risk control, checked first. Usually an
        #      unreverted, losing exit; that is correct and intended.
        #   2. mean-cross — long unwinds once z climbs back to z_exit, short is
        #      the mirror image.
        exit_reason = None
        if time_barrier_bars is not None and (t - entry_idx) >= time_barrier_bars:
            exit_reason = "time_barrier"
        elif (position == 1 and zscore[t] >= z_exit) or (
            position == -1 and zscore[t] <= z_exit
        ):
            exit_reason = "mean_cross"

        if exit_reason is not None and t + 1 < n:
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
                    "exit_reason": exit_reason,
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

    def _report(label: str, trades: pd.DataFrame) -> None:
        print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
        print(f"Number of trades: {len(trades)}")
        if trades.empty:
            return
        print("\nBy exit reason:")
        print(trades["exit_reason"].value_counts().to_string())
        print(f"\nBars held — mean: {trades['bars_held'].mean():.2f}, "
              f"median: {trades['bars_held'].median():.1f}")
        print(f"Total spread PnL: {trades['spread_pnl'].sum():.4f}")
        # Split mean PnL by exit reason — shows mean_cross winners against the
        # time_barrier (failure-to-revert) exits separately.
        print("\nMean spread PnL by exit reason:")
        print(trades.groupby("exit_reason")["spread_pnl"].mean().to_string())

    # Position-layer smoke test only — raw spread-PnL accounting, no costs.
    prices = load_basket_bars_cached(BASKET, "1h")
    spread = build_spread(prices, WEIGHTS)
    signals = generate_signals(spread, half_life_days=0.68)

    # (a) minimal: mean-cross exit only
    _report("MINIMAL (mean-cross exit only)", simulate_positions(signals))

    # (b) with time barrier at 5x half-life
    _report(
        "WITH TIME BARRIER (time_hl_mult=5.0)",
        simulate_positions(signals, half_life_days=0.68, freq="1h", time_hl_mult=5.0),
    )
