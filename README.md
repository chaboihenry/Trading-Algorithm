# Autonomous Algorithmic Trading Agent

A fully autonomous statistical arbitrage system built for Alpaca's paper trading environment. Deployed with $100K starting capital over a 90-day live trial.

## How It Works

The agent runs 24/7 across two nodes — a **Research Node** (MacBook Pro) that discovers opportunities overnight, and an **Execution Node** (ASUS Docker container) that trades them intraday via WebSocket.

**Signal Generation** — DBSCAN clusters correlated assets from a 110-ticker universe, then Johansen cointegration isolates mean-reverting spreads with statistically significant half-lives.

**Trade Filtering** — An XGBoost meta-labeler evaluates every signal using fractionally differentiated features and microstructure dynamics. Only setups exceeding a dynamic probability threshold reach the order router.

**Risk Control** — Hierarchical Risk Parity (HRP) allocates capital across active spreads daily, preventing hidden correlation blowups. A CUSUM filter monitors live tick flow for regime breaks and force-liquidates vulnerable positions.

## Data

Training data sourced from **Wharton Research Data Services (WRDS)** — TAQ millisecond-resolution trades across all U.S. exchanges, spanning **January 2021 through February 2026**.

## Tech Stack

Python · XGBoost · Numba · Statsmodels · Alpaca API · Docker · Git-based CI/CD between nodes

---

⚠️ **Disclaimer:** This project is actively in development. Architecture, models, and performance are subject to change. Not financial advice.
