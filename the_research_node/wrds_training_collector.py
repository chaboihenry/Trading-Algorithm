import os
import gc
import json
import time
import wrds
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

from the_utilities.paths import LOGS_DIR

# --- CONFIGURATION ---
WRDS_USERNAME = "henryvianna"
UNIVERSE_PATH = "universe.txt"
LOG_FILE = os.path.join(LOGS_DIR, "wrds_collection.log")

# Schema must match all existing parquet files
STRICT_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("exchange", pa.string()),
])

TICKER_BATCH_SIZE = 30
VAULT_ROOT = "/Volumes/Vault/quant_data/tick data storage"


def load_universe():
    with open(UNIVERSE_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]

def get_trading_days(start, end):
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days

def get_month_key(dt):
    return dt.strftime("%Y_%m")

def log(msg):
    print(msg)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")

def detect_last_collected_month(universe):
    # Scans the vault to find the most recent month file across all tickers
    # Returns the first day of the NEXT month to collect
    latest_month = None

    for ticker in universe:
        ticker_dir = os.path.join(VAULT_ROOT, ticker, "parquet", "training_data")
        if not os.path.exists(ticker_dir):
            continue

        for f in os.listdir(ticker_dir):
            if f.endswith('.parquet') and not f.startswith('._') and len(f) == 15:
                # Format: YYYY_MM.parquet
                month_str = f.replace('.parquet', '')
                try:
                    month_dt = datetime.strptime(month_str, "%Y_%m")
                    if latest_month is None or month_dt > latest_month:
                        latest_month = month_dt
                except ValueError:
                    continue

    if latest_month is None:
        log("[SYSTEM] No existing data found. Starting from 2021-01-01.")
        return datetime(2021, 1, 1)

    # Move to the first day of the next month
    if latest_month.month == 12:
        next_month = datetime(latest_month.year + 1, 1, 1)
    else:
        next_month = datetime(latest_month.year, latest_month.month + 1, 1)

    log(f"[SYSTEM] Latest data: {latest_month.strftime('%Y_%m')}. Resuming from {next_month.date()}.")
    return next_month

def query_single_day(db, date_str, ticker_batch):
    table_name = f"ctm_{date_str}"
    tickers_sql = ", ".join(f"'{t}'" for t in ticker_batch)

    query = f"""
        SELECT date, time_m, price, size, ex, sym_root
        FROM taqmsec.{table_name}
        WHERE sym_root IN ({tickers_sql})
        AND sym_suffix IS NULL
        AND tr_corr = '00'
        AND price > 0
    """
    try:
        df = db.raw_sql(query)
        return df
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "UndefinedTable" in error_msg:
            return pd.DataFrame()
        else:
            log(f"  [ERROR] Query failed for {table_name}: {error_msg}")
            return pd.DataFrame()

def clean_daily_batch(df):
    # Cleans and deduplicates the dataframe immediately to free RAM
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_m"].astype(str),
        format='mixed'
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")

    df = df.rename(columns={"ex": "exchange"})
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["exchange"] = df["exchange"].astype(str)

    df = df[["timestamp", "price", "size", "exchange", "sym_root"]]
    df = df.drop_duplicates(subset=["timestamp", "price", "size", "sym_root"], keep="first")
    df = df.sort_values(["sym_root", "timestamp"]).reset_index(drop=True)

    return df

def run_incremental_collection():
    # Detects the last collected month and fetches only new data
    log("====== WRDS TAQ INCREMENTAL COLLECTOR ======")

    universe = load_universe()
    log(f"Universe: {len(universe)} tickers")

    # Auto-detect where to resume
    start_date = detect_last_collected_month(universe)
    end_date = datetime.now() - timedelta(days=1)  # WRDS data has ~1 day lag

    if start_date >= end_date:
        log("[INFO] All available WRDS data already collected. Nothing to fetch.")
        return

    log(f"Collection range: {start_date.date()} -> {end_date.date()}")

    trading_days = get_trading_days(start_date, end_date)
    if not trading_days:
        log("[INFO] No trading days in range.")
        return

    # Group days by month
    months = {}
    for day in trading_days:
        key = get_month_key(day)
        if key not in months:
            months[key] = []
        months[key].append(day)

    log("[SYSTEM] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    log("[SYSTEM] Connected.")

    total_rows = 0

    for month_idx, (month_key, days) in enumerate(sorted(months.items()), 1):
        log(f"\n[{month_idx}/{len(months)}] {month_key}: {len(days)} trading days")

        # Open direct-to-disk writers for every ticker
        writers = {}
        for ticker in universe:
            ticker_dir = os.path.join(VAULT_ROOT, ticker, "parquet", "training_data")
            os.makedirs(ticker_dir, exist_ok=True)
            file_path = os.path.join(ticker_dir, f"{month_key}.parquet")
            writers[ticker] = pq.ParquetWriter(file_path, STRICT_SCHEMA, compression="zstd")

        month_rows = 0

        for day_idx, day in enumerate(days):
            date_str = day.strftime("%Y%m%d")

            for batch_start in range(0, len(universe), TICKER_BATCH_SIZE):
                batch = universe[batch_start: batch_start + TICKER_BATCH_SIZE]
                df = query_single_day(db, date_str, batch)

                if df.empty:
                    continue

                df = clean_daily_batch(df)

                for ticker in batch:
                    ticker_rows = df[df["sym_root"] == ticker]
                    if not ticker_rows.empty:
                        ticker_rows = ticker_rows[["timestamp", "price", "size", "exchange"]]
                        table = pa.Table.from_pandas(ticker_rows, schema=STRICT_SCHEMA)
                        writers[ticker].write_table(table)
                        month_rows += len(ticker_rows)

                del df
                gc.collect()

            if (day_idx + 1) % 5 == 0:
                log(f"    Day {day_idx + 1}/{len(days)} | Streaming to SSD...")

        # Close all writers for this month
        for ticker in universe:
            writers[ticker].close()

        total_rows += month_rows
        log(f"  [COMPLETED] {month_key}: {month_rows:,} rows written to disk")
        time.sleep(2)

    db.close()
    log(f"\n[DONE] Incremental collection complete. {total_rows:,} new rows processed.")

def run_full_rebuild(start_year: int = 2021):
    # Full collection from scratch — use when vault is empty or corrupted
    log(f"====== WRDS TAQ FULL REBUILD FROM {start_year} ======")

    universe = load_universe()
    log(f"Universe: {len(universe)} tickers")

    start_date = datetime(start_year, 1, 1)
    end_date = datetime.now() - timedelta(days=1)

    log(f"Collection range: {start_date.date()} -> {end_date.date()}")

    trading_days = get_trading_days(start_date, end_date)
    months = {}
    for day in trading_days:
        key = get_month_key(day)
        if key not in months:
            months[key] = []
        months[key].append(day)

    log("[SYSTEM] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    log("[SYSTEM] Connected.")

    total_rows = 0

    for month_idx, (month_key, days) in enumerate(sorted(months.items()), 1):
        log(f"\n[{month_idx}/{len(months)}] {month_key}: {len(days)} trading days")

        writers = {}
        for ticker in universe:
            ticker_dir = os.path.join(VAULT_ROOT, ticker, "parquet", "training_data")
            os.makedirs(ticker_dir, exist_ok=True)
            file_path = os.path.join(ticker_dir, f"{month_key}.parquet")
            writers[ticker] = pq.ParquetWriter(file_path, STRICT_SCHEMA, compression="zstd")

        month_rows = 0

        for day_idx, day in enumerate(days):
            date_str = day.strftime("%Y%m%d")

            for batch_start in range(0, len(universe), TICKER_BATCH_SIZE):
                batch = universe[batch_start: batch_start + TICKER_BATCH_SIZE]
                df = query_single_day(db, date_str, batch)

                if df.empty:
                    continue

                df = clean_daily_batch(df)

                for ticker in batch:
                    ticker_rows = df[df["sym_root"] == ticker]
                    if not ticker_rows.empty:
                        ticker_rows = ticker_rows[["timestamp", "price", "size", "exchange"]]
                        table = pa.Table.from_pandas(ticker_rows, schema=STRICT_SCHEMA)
                        writers[ticker].write_table(table)
                        month_rows += len(ticker_rows)

                del df
                gc.collect()

            if (day_idx + 1) % 5 == 0:
                log(f"    Day {day_idx + 1}/{len(days)} | Streaming to SSD...")

        for ticker in universe:
            writers[ticker].close()

        total_rows += month_rows
        log(f"  [COMPLETED] {month_key}: {month_rows:,} rows written to disk")
        time.sleep(2)

    db.close()
    log(f"\n[DONE] Full rebuild complete. {total_rows:,} rows processed.")


if __name__ == "__main__":
    run_incremental_collection()