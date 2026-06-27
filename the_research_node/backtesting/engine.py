"""
Generic, dollar-based, event-driven backtest engine.

Simulates ANY brain implementing the_utilities.strategy_interface.Strategy. The
engine is deliberately strategy-agnostic: it knows nothing about spreads,
z-scores, or cointegration. Each bar it feeds market_data to the strategy, acts
on the returned list[TradeIntent], and books fills/costs/PnL in DOLLARS.

Two conventions carry over verified from the spread-unit backtester layers:

  - NEXT-BAR FILL (cf. positions.py): an intent emitted from generate_targets at
    the close of bar t fills at bar t+1's price. Engine-side stop/horizon exits
    obey the same t -> t+1 timing. No same-bar lookahead.

  - ACTUAL-PRICE-AT-FILL COSTS (cf. costs.py): cost uses each asset's real price
    at ITS fill bar; cost = (bps/10000) * |dollar_notional_traded|, charged on
    both the entry fill and the exit fill.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from the_utilities.strategy_interface import Strategy, TradeIntent, Action

# Trailing window (bars) used to estimate each group's variance for the
# inverse-variance HRP stand-in. "Recent" history only — distant regimes should
# not dominate today's capital split.
_HRP_LOOKBACK = 60


@dataclass
class _Position:
    asset: str
    shares: float          # signed; >0 long, <0 short
    entry_price: float
    entry_bar: int
    group: str | None
    stop_level: float | None
    horizon_bars: int | None


def _periods_per_year(frequency: str) -> int:
    # Annualization factor for Sharpe, inferred from the strategy's cadence.
    # US equity calendar: ~252 trading days, ~7 RTH hours, ~13 half-hours.
    return {
        "daily": 252,
        "1h": 252 * 7,
        "30min": 252 * 13,
    }.get(frequency, 252)


def _allocate_capital_hrp(
    groups: dict,            # group_key -> list[TradeIntent] (the OPEN legs)
    history: pd.DataFrame,   # market_data through the decision bar (no lookahead)
    capital: float,
) -> dict:
    # Distribute capital ACROSS independent bets (groups). This is the seam where
    # full Hierarchical Risk Parity slots in later, shared with the live chassis.
    #
    # SIMPLIFIED PLACEHOLDER: a single bet takes everything (HRP no-op, correct
    # for one bet); multiple bets split by INVERSE VARIANCE of each bet's recent
    # synthetic-spread changes. Real HRP (clustering + recursive bisection)
    # replaces the body without changing this interface.
    if len(groups) <= 1:
        return {key: capital for key in groups}

    inv_var = {}
    for key, intents in groups.items():
        # Reconstruct the bet's "spread" as the weighted combination of its legs,
        # then measure variance of its recent bar-to-bar changes. diff (not
        # pct_change) because a signed spread can cross zero.
        series = None
        for intent in intents:
            weight = intent.target_weight or 0.0
            leg = history[intent.asset] * weight
            series = leg if series is None else series + leg
        changes = series.tail(_HRP_LOOKBACK).diff().dropna()
        var = float(changes.var()) if len(changes) else 0.0
        # Floor the variance so a (near-)constant bet cannot grab infinite weight.
        inv_var[key] = 1.0 / var if var > 0 else 0.0

    total_inv = sum(inv_var.values())
    if total_inv <= 0:
        # No usable variance signal -> fall back to equal weight across bets.
        share = capital / len(groups)
        return {key: share for key in groups}
    return {key: capital * inv_var[key] / total_inv for key in groups}


def run_backtest(
    strategy: Strategy,
    market_data: pd.DataFrame,   # columns = assets, index = DatetimeIndex, values = prices
    capital: float = 100_000.0,
    bps: float = 7.0,            # per-asset, per-side transaction cost
) -> dict:
    """Event-driven dollar backtest of `strategy` over `market_data`.

    Returns a results dict: trades, equity_curve, total_net_pnl,
    total_gross_pnl, total_cost, n_trades, sharpe, max_drawdown, turnover.
    """
    cost_rate = bps / 10_000.0
    n = len(market_data)
    start = strategy.required_history

    open_positions: dict[str, _Position] = {}  # keyed by asset; silence = hold
    entry_costs: dict[str, float] = {}          # entry-side cost held until close
    closed: list[dict] = []
    equity_times: list = []
    equity_values: list[float] = []

    cash = capital
    total_notional = 0.0  # absolute dollar volume across every fill (for turnover)

    for t in range(start, n):
        prices_t = market_data.iloc[t]
        ts = market_data.index[t]

        # MARK-TO-MARKET (recorded here, on bar-t state BEFORE this bar's fills):
        # positions decided this bar fill at t+1 and so belong to the next bar's
        # equity. Marking pre-fill keeps the equity curve free of the spurious
        # entry/exit-price jump that post-fill marking at price_t would inject.
        equity = cash + sum(
            pos.shares * prices_t[pos.asset] for pos in open_positions.values()
        )
        equity_times.append(ts)
        equity_values.append(equity)

        # 1. ENGINE-SIDE RISK (strategy-agnostic): stop / horizon breaches on
        #    existing positions, evaluated at prices_t, filled next bar.
        pending_exits: dict[str, str] = {}  # asset -> exit reason
        for asset, pos in open_positions.items():
            price = prices_t[asset]
            if pos.stop_level is not None and (
                (pos.shares > 0 and price <= pos.stop_level)
                or (pos.shares < 0 and price >= pos.stop_level)
            ):
                pending_exits[asset] = "stop"
                continue
            if pos.horizon_bars is not None and (t - pos.entry_bar) >= pos.horizon_bars:
                pending_exits[asset] = "horizon"

        # 2. ASK THE BRAIN with data through t only (structural no-lookahead).
        intents = strategy.generate_targets(market_data.iloc[: t + 1], open_positions)
        open_intents: list[TradeIntent] = []
        for intent in intents:
            if intent.action == Action.CLOSE:
                # CLOSE only matters for a position we actually hold.
                if intent.asset in open_positions:
                    pending_exits.setdefault(intent.asset, "strategy_close")
            elif intent.action == Action.OPEN:
                open_intents.append(intent)

        # 3. SIZE the OPEN intents via HRP across groups. group=None -> the intent
        #    is its own singleton bet keyed by asset.
        groups: dict = {}
        for intent in open_intents:
            key = intent.group if intent.group is not None else intent.asset
            groups.setdefault(key, []).append(intent)

        # Fills (sizing + execution) require a next bar to fill against.
        if t + 1 < n:
            fill_prices = market_data.iloc[t + 1]
            group_capital = _allocate_capital_hrp(
                groups, market_data.iloc[: t + 1], capital
            )

            # Within a bet, split its capital across legs by target_weight
            # proportions; the weight's SIGN sets long/short.
            sized: list[tuple] = []  # (asset, shares, group, stop, horizon)
            for key, legs in groups.items():
                gcap = group_capital.get(key, 0.0)
                abs_sum = sum(abs(it.target_weight) for it in legs if it.target_weight)
                if abs_sum == 0:
                    continue  # nothing actionable to size
                for it in legs:
                    if not it.target_weight:  # None or 0 -> not an OPEN we can size
                        continue
                    price = fill_prices[it.asset]
                    if pd.isna(price) or price <= 0:
                        continue
                    leg_dollars = gcap * abs(it.target_weight) / abs_sum
                    shares = np.sign(it.target_weight) * leg_dollars / price
                    sized.append(
                        (it.asset, shares, it.group, it.stop_level, it.horizon_bars)
                    )

            # 4. EXECUTE FILLS at t+1. Exits first so a same-bar close-then-reopen
            #    of one asset resolves cleanly.
            for asset, reason in pending_exits.items():
                pos = open_positions.pop(asset, None)
                if pos is None:
                    continue
                exit_price = fill_prices[asset]
                exit_cost = cost_rate * abs(pos.shares * exit_price)
                gross_pnl = pos.shares * (exit_price - pos.entry_price)
                cost = entry_costs.pop(asset, 0.0) + exit_cost
                cash += pos.shares * exit_price - exit_cost
                total_notional += abs(pos.shares * exit_price)
                closed.append(
                    {
                        "asset": asset,
                        "group": pos.group,
                        "entry_time": market_data.index[pos.entry_bar],
                        "exit_time": market_data.index[t + 1],
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "shares": pos.shares,
                        "gross_pnl": gross_pnl,
                        "cost": cost,
                        "net_pnl": gross_pnl - cost,
                        "exit_reason": reason,
                    }
                )

            # New opens. An asset already held is left untouched (silence = hold;
            # brains emit intents for CHANGES only).
            for asset, shares, group, stop, horizon in sized:
                if asset in open_positions:
                    continue
                entry_price = fill_prices[asset]
                entry_cost = cost_rate * abs(shares * entry_price)
                cash -= shares * entry_price + entry_cost
                total_notional += abs(shares * entry_price)
                open_positions[asset] = _Position(
                    asset, shares, entry_price, t + 1, group, stop, horizon
                )
                entry_costs[asset] = entry_cost

    # Force-close anything still open at the last bar.
    if n > 0:
        last = n - 1
        last_prices = market_data.iloc[last]
        for asset, pos in list(open_positions.items()):
            exit_price = last_prices[asset]
            exit_cost = cost_rate * abs(pos.shares * exit_price)
            gross_pnl = pos.shares * (exit_price - pos.entry_price)
            cost = entry_costs.pop(asset, 0.0) + exit_cost
            cash += pos.shares * exit_price - exit_cost
            total_notional += abs(pos.shares * exit_price)
            closed.append(
                {
                    "asset": asset,
                    "group": pos.group,
                    "entry_time": market_data.index[pos.entry_bar],
                    "exit_time": market_data.index[last],
                    "entry_price": pos.entry_price,
                    "exit_price": exit_price,
                    "shares": pos.shares,
                    "gross_pnl": gross_pnl,
                    "cost": cost,
                    "net_pnl": gross_pnl - cost,
                    "exit_reason": "end_of_data",
                }
            )
            del open_positions[asset]

    # The per-bar equity samples are marked PRE-FILL, so the last sample (bar
    # n-1) still carries the force-closed positions at their mark and never sees
    # the closing fills booked above. With every position now flat, realized
    # terminal equity is simply `cash`; overwrite the final point so the curve
    # reconciles with total_net_pnl (no-op when nothing was force-closed).
    if equity_values:
        equity_values[-1] = cash

    trade_columns = [
        "asset", "group", "entry_time", "exit_time", "entry_price", "exit_price",
        "shares", "gross_pnl", "cost", "net_pnl", "exit_reason",
    ]
    trades = pd.DataFrame(closed, columns=trade_columns)
    equity_curve = pd.Series(equity_values, index=equity_times, dtype=float)

    # Sharpe on equity-curve returns, annualized by the strategy's cadence.
    rets = equity_curve.pct_change(fill_method=None).dropna()
    if len(rets) < 2 or rets.std() == 0:
        sharpe = 0.0
    else:
        ppy = _periods_per_year(strategy.frequency)
        sharpe = float(np.sqrt(ppy) * rets.mean() / rets.std())

    # Max peak-to-trough drawdown as a positive fraction of the running peak.
    if len(equity_curve):
        eq = equity_curve.to_numpy()
        peak = np.maximum.accumulate(eq)
        drawdown = np.where(peak > 0, (eq - peak) / peak, 0.0)
        max_drawdown = float(-drawdown.min())
    else:
        max_drawdown = 0.0

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "total_net_pnl": float(trades["net_pnl"].sum()),
        "total_gross_pnl": float(trades["gross_pnl"].sum()),
        "total_cost": float(trades["cost"].sum()),
        "n_trades": int(len(trades)),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": total_notional / capital,
    }


def breakeven_bps(
    strategy: Strategy,
    market_data: pd.DataFrame,
    capital: float = 100_000.0,
) -> float:
    """Per-side bps at which net PnL hits zero.

    Cost is linear in bps and gross PnL is bps-invariant, so running once at a
    1-bp reference gives cost-per-bp directly; the break-even is just the ratio
    total_gross_pnl / cost_at_1bp (scale-invariant).
    """
    res = run_backtest(strategy, market_data, capital=capital, bps=1.0)
    cost_at_1bp = res["total_cost"]
    if cost_at_1bp == 0:
        return float("inf")  # no traded notional -> costs never bite
    return res["total_gross_pnl"] / cost_at_1bp
