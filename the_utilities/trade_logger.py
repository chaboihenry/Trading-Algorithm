import os
import csv
from threading import Lock

from the_utilities.paths import EXECUTION_DATA_DIR, TRADE_HISTORY_CSV


# Thread-safe writer lock so streamer and order threads can both append safely
_write_lock = Lock()

# Schema: any key missing from trade_data will write as empty string in its column
CSV_HEADERS = [
    "exit_timestamp",
    "spread_name",
    "direction",
    "entry_timestamp",
    "bars_held",
    "exit_reason",
    "entry_z",
    "exit_z",
    "ai_confidence",
    "bet_size",
    "capital_allocated",
    "tickers",
    "entry_prices",
    "exit_prices",
    "shares",
    "gross_pnl",
    "slippage_cost",
    "net_pnl",
    "pnl_pct",
]


def _ensure_csv_exists():
    # Create the data dir and CSV with header row if either is missing
    os.makedirs(EXECUTION_DATA_DIR, exist_ok=True)
    if not os.path.exists(TRADE_HISTORY_CSV):
        with open(TRADE_HISTORY_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def log_trade(trade_data: dict):
    # Append one closed-trade record; missing keys silently become empty cells
    with _write_lock:
        _ensure_csv_exists()
        with open(TRADE_HISTORY_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trade_data.get(col, "") for col in CSV_HEADERS])