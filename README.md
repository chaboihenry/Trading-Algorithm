# Integrated Algorithmic Trading Agent

An automated, end-to-end algorithmic trading system engineered for the Alpaca brokerage. This agent leverages unsupervised machine learning, advanced meta-labeling, and tick-level regime detection to execute and manage intraday statistical arbitrage strategies while strictly controlling downside risk and portfolio concentration.

## System Architecture

The agent operates through four highly integrated, specialized modules:

### 1. The Brain: Intraday Cointegration Arbitrage
The primary signal generator for long/short pairs trading. It avoids the standard multiple-testing trap by employing a two-step dimensionality reduction pipeline:
* **Search Space Reduction:** Utilizes unsupervised ML (DBSCAN) to identify highly correlated asset clusters.
* **Cointegration Verification:** Applies GPU-accelerated (cuDF) cointegration testing across the clustered universe using high-frequency tick data to isolate statistically robust trading pairs.

### 2. The Filter: Meta-Labeling (GPU-Accelerated XGBoost)
Primary signals are not traded blindly. Once a signal (e.g., "Buy NVDA") is generated, the Meta-Labeler evaluates it using fractionally differentiated features to preserve memory without sacrificing stationarity. It calculates the exact probability of the specific setup being profitable. The trade is only passed to the execution queue if the probability exceeds a strict dynamic threshold.

### 3. The Shield: Regime Detection (CUSUM Filter)
The system's emergency brake. While holding live positions, the Shield continuously monitors incoming tick data for structural breaks. If order flow toxicity spikes or a sudden regime shift occurs, the CUSUM filter trips, overriding the primary logic and immediately liquidating the vulnerable positions to cash to prevent catastrophic drawdowns.

### 4. The Allocator: Hierarchical Risk Parity (HRP)
At the daily close, the Allocator processes the total available cash pool. It applies Hierarchical Risk Parity (HRP) to calculate the optimal capital allocation for the next trading session across the active universe (e.g., AAPL, NVDA, TSLA). This ensures that hidden correlations do not lead to concentrated risk exposure.
