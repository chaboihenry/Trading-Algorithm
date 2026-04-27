import os
import csv
from threading import Lock


_write_lock = Lock()

LOG_DIR = "the_execution_node/data"
LOG_FILE = os.path.join(LOG_DIR, "trade_history.csv")

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
    # Create the CSV with headers if it doesn't exist yet
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADERS)


def log_trade(trade_data: dict):
    # Append a single trade record to the CSV
    with _write_lock:
        _ensure_csv_exists()
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trade_data.get(col, "") for col in CSV_HEADERS])