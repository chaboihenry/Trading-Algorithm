"""
Stat-arb backtester — LAYER 4: transaction-cost model.

Applies a fixed per-leg, per-side bps cost to the closed trades from Layer 3
and computes (a) net PnL after costs and (b) the exact break-even bps at which
net PnL = 0.

Everything stays in Johansen-weighted SPREAD units. The spread is
sum_i w_i * price_i, so a cost expressed as (bps/10000) * sum_i |w_i| * price_i
is already in the same units as spread_pnl and subtracts directly — no
notional/return conversion needed.
"""

import pandas as pd


def _gross_notional(prices: pd.Series, weights: dict) -> float:
    # Cost basis of trading the basket once: every leg contributes its
    # absolute weighted price, regardless of long/short sign.
    return sum(abs(weights[t]) * prices[t] for t in weights)


def apply_costs(
    trades: pd.DataFrame,  # from simulate_positions
    panel: pd.DataFrame,   # the per-leg hourly price panel (11 cols)
    weights: dict,         # WEIGHTS
    bps: float = 7.0,      # per-leg, per-side
) -> pd.DataFrame:
    out = trades.copy()
    gross_entry = []
    gross_exit = []
    for _, tr in out.iterrows():
        # Use the ACTUAL per-leg prices at each fill bar — entry and exit
        # notionals differ as prices move, so a single snapshot would misprice.
        entry_prices = panel.iloc[int(tr["entry_idx"])]
        exit_prices = panel.iloc[int(tr["exit_idx"])]
        gross_entry.append(_gross_notional(entry_prices, weights))
        gross_exit.append(_gross_notional(exit_prices, weights))

    out["gross_notional_entry"] = gross_entry
    out["gross_notional_exit"] = gross_exit
    # Round-trip = every leg traded twice (entry + exit).
    out["trade_cost"] = (bps / 10000.0) * (
        out["gross_notional_entry"] + out["gross_notional_exit"]
    )
    out["net_pnl"] = out["spread_pnl"] - out["trade_cost"]
    return out


def breakeven_bps(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    weights: dict,
) -> float:
    # Cost is linear in bps, so the break-even is simply total gross PnL
    # divided by the total cost at 1 bp — no root-finding required.
    cost_at_1bp = apply_costs(trades, panel, weights, bps=1.0)["trade_cost"].sum()
    if cost_at_1bp == 0:
        return float("inf")  # no traded notional -> never eaten by costs
    return trades["spread_pnl"].sum() / cost_at_1bp


if __name__ == "__main__":
    from the_research_node.backtesting.data import (
        load_basket_bars_cached,
        build_spread,
        BASKET,
        WEIGHTS,
    )
    from the_research_node.backtesting.signals import generate_signals
    from the_research_node.backtesting.positions import simulate_positions

    # Cost-layer smoke test — net spread-PnL accounting against a bps grid.
    panel = load_basket_bars_cached(BASKET, "1h")
    spread = build_spread(panel, WEIGHTS)
    signals = generate_signals(spread, half_life_days=0.68)
    trades = simulate_positions(
        signals, half_life_days=0.68, freq="1h", time_hl_mult=5.0
    )

    priced = apply_costs(trades, panel, WEIGHTS, bps=7.0)
    print(f"Trades: {len(priced)}")
    print(f"Total gross spread PnL: {priced['spread_pnl'].sum():.4f}")
    print(f"Total trade cost (7 bps): {priced['trade_cost'].sum():.4f}")
    print(f"Total net PnL (7 bps):    {priced['net_pnl'].sum():.4f}")
    if not priced.empty:
        print(f"Mean net PnL per trade:   {priced['net_pnl'].mean():.4f}")
        print(f"Net-positive trades: {int((priced['net_pnl'] > 0).sum())}")
        print(f"Net-negative trades: {int((priced['net_pnl'] < 0).sum())}")

    be = breakeven_bps(trades, panel, WEIGHTS)
    print(f"\nBreak-even bps: {be:.4f}")

    # Net PnL across a bps grid for context — strategy survives while the grid
    # value stays below break-even.
    print("\nTotal net PnL by per-leg-per-side bps:")
    for bps in [1, 2, 3, 5, 7, 10]:
        net = apply_costs(trades, panel, WEIGHTS, bps=float(bps))["net_pnl"].sum()
        print(f"  {bps:>2} bps: {net:.4f}")
