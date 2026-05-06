# Signal thresholds
Z_THRESH = 2.39
AI_THRESH = 0.56

# Triple-barrier exit thresholds (multipliers on per-bar volatility)
PT_SKEW = 1.90
SL_SKEW = 1.75
TIME_BARRIER = 120

# Capital parameters
LEVERAGE = 2.0
HRP_MAX_CAP = 0.25
HRP_MIN_FLOOR = 0.03
STARTING_EQUITY = 100_000.0

# Cost model — split for honest analytics
SLIPPAGE_BPS_BACKTEST = 7.0
SLIPPAGE_BPS_LIVE = 2.5
SHORT_BORROW_APR = 0.01

# Risk management
COOLDOWN_MINUTES = 30
EOD_LIQUIDATION_TIME_MINUTES = 950  # 15:50 ET
SAFE_ENTRY_WINDOW = (585, 945)  # 09:45 to 15:45 ET, in minutes-since-midnight
EOD_COOLDOWN_SKIP_MINUTES = 920  # 15:20 — skip cooldown checks within 30 min of EOD
CUSUM_REGIME_THRESHOLD = 0.02
CUSUM_RECOVERY_DAYS = 3

# Data
WARMUP_BARS = 2340
BARS_PER_YEAR = 252 * 78

COOLDOWN_MINUTES = 30
COOLDOWN_BARS = 6  # COOLDOWN_MINUTES / 5min per bar — for backtester

# Trading constraints
NON_SHORTABLE_TICKERS = {'SO'}  # Tickers Alpaca doesn't allow shorting
COOLDOWN_BARS = COOLDOWN_MINUTES // 5  # Derived: 30 minutes / 5min per bar = 6

HEDGE_RATIO_MIN = 0.15  # Minimum ratio of smallest-to-largest leg notional