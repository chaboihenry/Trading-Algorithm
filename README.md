# Statistical Arbitrage Research & Execution Framework

**Headline result:** Cointegration-based statistical arbitrage on liquid US
equities and ETFs has **no tradeable edge after transaction costs** in the
2021–2026 regime. This repository is the rigorous research-and-execution
pipeline that demonstrated it — and a modular framework built so a different
trading strategy can replace the cointegration "brain" without rewriting the
execution engine.

The most valuable artifact here is not a profitable bot. It is an honest
backtester that, after the strategy was built end to end, proved the edge does
not survive costs — and the discipline to accept that result rather than tune
the numbers until they looked good.

---
## The findings

Tested universe: **148 liquid US tickers** (a mix of individual equities and
ETFs), ~5.5 years of trade data (2021–2026) collected from WRDS at the tick
level (~2.1 billion rows), split-adjusted via CRSP cumulative factors.

The results form a gradient, from "no signal" to "signal but untradeable":

**1. Individual-equity pairs do not cointegrate (2021–2026).**
Classic candidate pairs across financials, retail, and payments (e.g. BAC/JPM,
HD/LOW, MA/V) failed cointegration tests across rolling windows despite high
correlation. This reproduces the post-2010 decay of equity pairs trading
documented by Krauss (2017).

**2. Some ETF pairs cointegrate, but only conditionally.**
HYG/TLT (high-yield credit vs. long Treasuries) was cointegrated in only ~34%
of rolling 252-day windows — a regime-dependent relationship that would require
a live timing layer, not a static always-on pair.

**3. One ETF basket cointegrates robustly — and still has no tradeable edge.**
An 11-leg fixed-income basket (AGG, BND, EMB, HYG, HYLB, IEF, JNK, LQD, SHY,
TIP, TLT) was robustly cointegrated. But a from-scratch, lookahead-verified
backtester showed it is **untradeable after costs**:

| Metric | Value |
|---|---|
| Trades (5.5y, hourly, RTH only) | 491 |
| Gross edge per trade | ~0.30 spread-units |
| **Break-even cost** | **0.30 bps per leg, per side** |
| Net PnL at 7 bps | −3,268 (only 2 of 491 trades profitable) |
| Net PnL at 1 bp (optimistic) | −340 (still a loss) |

The break-even cost of **0.30 bps** is roughly an order of magnitude below
realistic transaction costs for an 11-leg basket (even liquid bond ETFs carry
wider effective spreads, and the thinner legs much more). The strategy loses
money at *any* plausible cost level.

**Why:** the near-redundancy that makes these bond ETFs robustly cointegrated
(they share the same underlying rate and credit factors) is the same thing that
makes the spread barely move. Strong cointegration and a large tradeable edge
are in tension — the more reliably a basket reverts, the smaller the edge there
is to capture. This is the core lesson of the project.

All precise figures above are reproducible: run
`uv run python -m the_research_node.backtesting.costs`.

---
## The system

**Modular brain-chassis architecture.** The execution engine (the "chassis")
depends only on an abstract `Strategy` interface, never on a concrete strategy.
A strategy (the "brain") implements `generate_targets()` and returns a list of
`TradeIntent` objects (OPEN/CLOSE, target weight, optional stop and time
horizon, plus a group tag for risk allocation). Swapping strategies — e.g.
replacing cointegration with a momentum or news-sentiment model — means adding
one new brain module and a config string, with no changes to the execution code.
Each brain owns its own model artifacts under `the_models/<brain>/`.

**From-scratch backtester, built for honesty.** A prior backtester was deleted
for producing exaggerated results; this one was rebuilt layer by layer, each
layer verified against a specific source of inflation:
- data: real RTH-only bars (DST-aware), no overnight forward-fill artifacts
- signals: point-in-time z-scores, verified lookahead-free (future-deletion test)
- positions: next-bar fills (no same-bar lookahead), time-barrier exits
- costs: realistic per-leg per-side costs, plus exact break-even analysis

A second, **generic event-driven engine** (`backtesting/engine.py`) simulates
*any* `Strategy`'s `TradeIntent`s in dollar terms, through the same interface
the live chassis uses — so a strategy is backtested through its real execution
path.

**Research pipeline.** Tick-data collection (WRDS), split adjustment (CRSP),
DBSCAN cluster discovery, Johansen cointegration tests, rolling-window
validation, and Hierarchical Risk Parity allocation (López de Prado).

**Two-node deployment.** A research node (offline: data, discovery, training)
pushes curated artifacts to GitHub; an execution node (AWS EC2, Docker) pulls
them and trades a paper account on Alpaca.

---
## Engineering practices

- **Honest backtesting over optimistic backtesting.** A backtester is only
  valuable if you can trust it when it reports bad numbers. This one was
  deliberately built to that standard, and its headline result is negative.
- **Systematic issue tracking.** `issues.txt` records every bug found across the
  pipeline — severity, status, and resolution — including issues correctly
  identified as needing no code change. Detail lives in the git history.
- **Fail loud, never silently fall back.** A recurring bug class in this project
  was silent fallback to a wrong default path; the fix pattern throughout is to
  require configuration explicitly and fail loudly when it is missing.
- **Multiple-testing discipline.** With 148 tickers (thousands of candidate
  pairs), some cointegration "passes" are false positives by chance; every
  candidate must survive rolling-window validation, not a single test
  (Bailey & López de Prado).

---
## Status

- **Research pipeline + honest backtester:** complete. The negative finding
  above is the result.
- **Execution chassis correctness:** complete
- **Modular brain-chassis refactor (in progress):** the `Strategy` interface and
  the generic dollar-based engine are built; the engine's mechanics (next-bar
  fills, PnL, costs, equity curve) are validated on a buy-and-hold case against
  an independent hand computation. Next: migrate the cointegration logic to
  implement the interface, then validate the generic engine end-to-end by
  reproducing the 0.30-bps break-even result above. Remaining work is tracked in
  `issues.txt` (OPEN section).

This project is paused at this checkpoint. The framework is designed so that
testing a *new* strategy is the natural next step: implement a new brain against
the `Strategy` interface, backtest it through the generic engine, and deploy it
through the unchanged chassis.

---
## References

López de Prado, *Advances in Financial Machine Learning*. Krauss (2017),
*Statistical Arbitrage Pairs Trading Strategies*. Johansen (1991),
cointegration. Bailey & López de Prado, deflated Sharpe ratio.
