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

# --- CONFIGURATION ---
WRDS_USERNAME = "henryvianna"
UNIVERSE_PATH = "universe.txt"
LOG_FILE = "logs/wrds_collection.log"

# 5 years back from the latest available table (Feb 2026)
START_DATE = datetime(2021, 3, 1)
END_DATE = datetime(2026, 2, 24)

# Strict schema for consistency across all files
STRICT_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("exchange", pa.string()),
])

# How many tickers to query per SQL call to avoid WRDS timeouts
TICKER_BATCH_SIZE = 30


def load_universe():
    # Reads the 113 symbols from universe.txt
    with open(UNIVERSE_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_trading_days(start, end):
    # Generates all potential trading days (Mon-Fri) in the range
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Mon=0, Fri=4
            days.append(current)
        current += timedelta(days=1)
    return days


def get_month_key(dt):
    # Returns YYYY_MM string for grouping days into monthly files
    return dt.strftime("%Y_%m")


def log(msg):
    # Dual logging — terminal + file
    print(msg)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def build_timestamp(df):
    # Combines WRDS 'date' + 'time_m' into a proper UTC nanosecond timestamp
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_m"].astype(str)
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")
    return df


def query_single_day(db, date_str, ticker_batch):
    # Pulls all trades for a batch of tickers from one daily ctm_ table.
    # Filters: regular trades only (tr_corr='00'), no options (sym_suffix IS NULL), price > 0
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


def flush_month(ticker, month_key, rows):
    # Writes a month's worth of trades to a single parquet file.
    if not rows:
        return 0

    df = pd.concat(rows, ignore_index=True)

    df = build_timestamp(df)
    df = df.rename(columns={"ex": "exchange"})
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["exchange"] = df["exchange"].astype(str)

    df = df[["timestamp", "price", "size", "exchange"]]

    df = df.drop_duplicates(subset=["timestamp", "price", "size"], keep="first")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # New Directory Structure
    ticker_dir = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"
    os.makedirs(ticker_dir, exist_ok=True)
    file_path = os.path.join(ticker_dir, f"{month_key}.parquet")

    table = pa.Table.from_pandas(df, schema=STRICT_SCHEMA)
    pq.write_table(table, file_path, compression="zstd")

    row_count = len(df)
    del df, table
    gc.collect()

    return row_count


def run_collection():
    log("====== WRDS TAQ TRAINING DATA COLLECTOR ======")
    log(f"Range: {START_DATE.date()} -> {END_DATE.date()}")
    
    universe = load_universe()
    log(f"Universe: {len(universe)} tickers")

    trading_days = get_trading_days(START_DATE, END_DATE)
    log(f"Trading days to process: {len(trading_days)}")

    months = {}
    for day in trading_days:
        key = get_month_key(day)
        if key not in months:
            months[key] = []
        months[key].append(day)

    log(f"Total months: {len(months)}")

    log("[SYSTEM] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    log("[SYSTEM] Connected.")

    total_rows = 0
    months_completed = 0

    for month_idx, (month_key, days) in enumerate(sorted(months.items()), 1):
        log(f"\n[{month_idx}/{len(months)}] {month_key}: {len(days)} trading days")

        month_buffer = {t: [] for t in universe}

        for day_idx, day in enumerate(days):
            date_str = day.strftime("%Y%m%d")

            for batch_start in range(0, len(universe), TICKER_BATCH_SIZE):
                batch = universe[batch_start : batch_start + TICKER_BATCH_SIZE]
                df = query_single_day(db, date_str, batch)

                if df.empty:
                    continue

                for ticker in batch:
                    ticker_rows = df[df["sym_root"] == ticker]
                    if not ticker_rows.empty:
                        month_buffer[ticker].append(ticker_rows)

                del df
                gc.collect()

            if (day_idx + 1) % 5 == 0:
                buffered = sum(len(r) for rows in month_buffer.values() for r in rows)
                log(f"    Day {day_idx + 1}/{len(days)} | Buffer: {buffered:,} rows")

        month_rows = 0
        for ticker in universe:
            rows = month_buffer[ticker]
            if rows:
                count = flush_month(ticker, month_key, rows)
                month_rows += count

        total_rows += month_rows
        months_completed += 1

        del month_buffer
        gc.collect()

        log(f"  [FLUSHED] {month_key}: {month_rows:,} rows across {len(universe)} tickers")
        time.sleep(2)

    db.close()

    log(f"\n{'=' * 60}")
    log(f"COLLECTION COMPLETE")
    log(f"{'=' * 60}")
    log(f"Months processed: {months_completed}/{len(months)}")
    log(f"Total rows collected: {total_rows:,}")


if __name__ == "__main__":
    run_collection()