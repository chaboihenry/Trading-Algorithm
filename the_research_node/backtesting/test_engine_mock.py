"""
Engine validation via the simplest possible brain: buy one asset, hold to end.

Proves the generic engine's mechanics (next-bar fill, dollar costs, force-close)
on a strategy with NO model, NO signal, and exactly one trade — so the expected
PnL can be recomputed by hand and matched against the engine's output. Doubles
as the minimal worked example of implementing the Strategy ABC.

  uv run python -m the_research_node.backtesting.test_engine_mock
"""

from the_utilities.strategy_interface import Strategy, TradeIntent, Action


class BuyAndHoldStrategy(Strategy):
    # Emits ONE OPEN intent for a single asset on its first decision, then
    # stays silent (silence = hold). Simplest possible Strategy.

    def __init__(self, asset: str):
        self._asset = asset
        self._opened = False

    @property
    def name(self) -> str:
        return "buy_and_hold"

    @property
    def frequency(self) -> str:
        return "1h"

    @property
    def required_history(self) -> int:
        return 1

    @property
    def model_dir(self) -> str:
        return ""  # mock has no artifacts

    def generate_targets(self, market_data, positions):
        if self._opened:
            return []  # hold forever
        self._opened = True
        return [TradeIntent(action=Action.OPEN, asset=self._asset,
                            target_weight=1.0, group="BH")]


if __name__ == "__main__":
    from the_research_node.backtesting.data import load_basket_bars_cached, BASKET
    from the_research_node.backtesting.engine import run_backtest

    panel = load_basket_bars_cached(BASKET, "1h")
    asset = "HYG"
    capital = 100_000.0
    bps = 7.0
    strat = BuyAndHoldStrategy(asset)
    res = run_backtest(strat, panel, capital=capital, bps=bps)

    # The mock opens on its first decision bar (t = required_history = 1),
    # which FILLS at bar t+1 = 2. It is force-closed at the last bar.
    entry_bar = 2
    last_bar = len(panel) - 1
    entry_price = panel[asset].iloc[entry_bar]
    exit_price = panel[asset].iloc[last_bar]
    cost_rate = bps / 10_000.0
    shares = capital / entry_price          # 100% weight, long
    exp_entry_cost = cost_rate * abs(shares * entry_price)
    exp_exit_cost = cost_rate * abs(shares * exit_price)
    exp_gross = shares * (exit_price - entry_price)
    exp_cost = exp_entry_cost + exp_exit_cost
    exp_net = exp_gross - exp_cost

    print("=== ENGINE RESULTS ===")
    print(f"n_trades:        {res['n_trades']}")
    print(f"total_gross_pnl: {res['total_gross_pnl']:.2f}")
    print(f"total_cost:      {res['total_cost']:.2f}")
    print(f"total_net_pnl:   {res['total_net_pnl']:.2f}")
    if res['n_trades'] == 1:
        tr = res['trades'].iloc[0]
        print(f"  entry_price: {tr['entry_price']:.4f}  exit_price: {tr['exit_price']:.4f}")
        print(f"  shares:      {tr['shares']:.4f}  exit_reason: {tr['exit_reason']}")
    print("\n=== INDEPENDENT EXPECTED ===")
    print(f"entry_bar={entry_bar} price={entry_price:.4f} | last_bar={last_bar} price={exit_price:.4f}")
    print(f"shares:     {shares:.4f}")
    print(f"exp_gross:  {exp_gross:.2f}")
    print(f"exp_cost:   {exp_cost:.2f}")
    print(f"exp_net:    {exp_net:.2f}")
    print("\n=== MATCH CHECKS ===")
    print(f"trades == 1:        {res['n_trades'] == 1}")
    print(f"gross matches:      {abs(res['total_gross_pnl'] - exp_gross) < 1.0}")
    print(f"cost matches:       {abs(res['total_cost'] - exp_cost) < 0.5}")
    print(f"net matches:        {abs(res['total_net_pnl'] - exp_net) < 1.0}")

    # Bug-exposure check: does the equity curve's last point equal
    # capital + net_pnl? (If the final force-close cost isn't reflected in the
    # curve, this reveals the gap and its exact size.)
    eq_last = res['equity_curve'].iloc[-1]
    print(f"\nequity_curve[-1]:        {eq_last:.2f}")
    print(f"capital + net_pnl:       {capital + res['total_net_pnl']:.2f}")
    print(f"difference:              {eq_last - (capital + res['total_net_pnl']):.2f}")
    print(f"(nonzero difference => final close not reflected in equity curve)")
