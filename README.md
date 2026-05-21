# Autonomous Algorithmic Trading Agent

A fully autonomous statistical arbitrage system trading in Alpaca's paper environment. Currently deployed on AWS EC2 with $100K starting capital for a 90-day live forward test.

## Current Status

🟢 Paper Trading Active — The agent is live on Alpaca's paper account. No validated out-of-sample results yet.

## How It Works

The system runs 24/7 across two nodes: a **Research Node** (MacBook Pro) that rediscovers opportunities after hours, and an **Execution Node** (Dockerized on AWS EC2) that trades them intraday via Alpaca's WebSocket.

**Signal Generation** — DBSCAN clusters correlated assets from a 110-ticker universe, then Johansen cointegration isolates mean-reverting spreads with statistically significant half-lives.

**Trade Filtering** — An XGBoost meta-labeler scores every candidate signal using fractionally differentiated features and microstructure dynamics. Only setups exceeding a dynamic probability threshold reach the order router.

**Risk Control** — Hierarchical Risk Parity (HRP) allocates capital across active spreads daily. A CUSUM filter on SPY monitors for regime breaks and blocks new entries during macro instability. Cooldown timers, EOD liquidation, and short-borrow checks prevent whipsaw and overnight gap risk.

## Data

Training data sourced from **Wharton Research Data Services (WRDS)** — TAQ millisecond-resolution trades across all U.S. exchanges, January 2021 through February 2026.

## Tech Stack

Python · XGBoost · Numba · Statsmodels · Alpaca API · Docker · AWS EC2 · Tailscale · Git-based sync between nodes

---

⚠️ **Disclaimer:** This project is actively in development and runs exclusively on paper capital. Architecture, models, and performance are subject to change. Not financial advice.
