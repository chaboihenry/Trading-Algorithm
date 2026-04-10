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

# PEPE: Resuming exactly at January 2022 to save you 3 hours of re-downloading
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2026, 2, 24)

# PEPE: Strict schema for consistency across all files
STRICT_SCHEMA = pa.schema([
    ("timestamp", pa.timestamp("ns", tz="UTC")),
    ("price", pa.float64()),
    ("size", pa.float64()),
    ("exchange", pa.string()),
])

TICKER_BATCH_SIZE = 30


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
    # Cleans and deduplicates the dataframe immediately to free up RAM.
    # format='mixed' protects against flat-second timestamp crashes
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time_m"].astype(str),
        format='mixed'
    )
    df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern").dt.tz_convert("UTC")

    df = df.rename(columns={"ex": "exchange"})
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    df["exchange"] = df["exchange"].astype(str)

    # Keep sym_root temporarily so we can sort and group
    df = df[["timestamp", "price", "size", "exchange", "sym_root"]]
    
    # Deduplicate and sort chronologically for the day
    df = df.drop_duplicates(subset=["timestamp", "price", "size", "sym_root"], keep="first")
    df = df.sort_values(["sym_root", "timestamp"]).reset_index(drop=True)
    
    return df


def run_collection():
    log("====== WRDS TAQ DIRECT-TO-DISK COLLECTOR ======")
    log(f"Range: {START_DATE.date()} -> {END_DATE.date()}")
    
    universe = load_universe()
    log(f"Universe: {len(universe)} tickers")

    trading_days = get_trading_days(START_DATE, END_DATE)
    months = {}
    for day in trading_days:
        key = get_month_key(day)
        if key not in months: months[key] = []
        months[key].append(day)

    log("[SYSTEM] Connecting to WRDS...")
    db = wrds.Connection(wrds_username=WRDS_USERNAME)
    log("[SYSTEM] Connected.")

    total_rows = 0
    months_completed = 0

    for month_idx, (month_key, days) in enumerate(sorted(months.items()), 1):
        log(f"\n[{month_idx}/{len(months)}] {month_key}: {len(days)} trading days")

        # 1. Open direct-to-disk pipelines for every ticker for this month
        writers = {}
        for ticker in universe:
            ticker_dir = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"
            os.makedirs(ticker_dir, exist_ok=True)
            file_path = os.path.join(ticker_dir, f"{month_key}.parquet")
            writers[ticker] = pq.ParquetWriter(file_path, STRICT_SCHEMA, compression="zstd")

        month_rows = 0

        # 2. Query, clean, write, and delete day-by-day
        for day_idx, day in enumerate(days):
            date_str = day.strftime("%Y%m%d")

            for batch_start in range(0, len(universe), TICKER_BATCH_SIZE):
                batch = universe[batch_start : batch_start + TICKER_BATCH_SIZE]
                df = query_single_day(db, date_str, batch)

                if df.empty:
                    continue

                # Clean the batch immediately to drop string overhead
                df = clean_daily_batch(df)

                for ticker in batch:
                    ticker_rows = df[df["sym_root"] == ticker]
                    if not ticker_rows.empty:
                        # Drop the sym_root column so it strictly matches the Parquet Schema
                        ticker_rows = ticker_rows[["timestamp", "price", "size", "exchange"]]
                        
                        # Write directly to the SSD
                        table = pa.Table.from_pandas(ticker_rows, schema=STRICT_SCHEMA)
                        writers[ticker].write_table(table)
                        month_rows += len(ticker_rows)

                # Annihilate the dataframe from RAM
                del df
                gc.collect()

            if (day_idx + 1) % 5 == 0:
                log(f"    Day {day_idx + 1}/{len(days)} | Streaming data to SSD...")

        # 3. Securely close all files at the end of the month
        for ticker in universe:
            writers[ticker].close()

        total_rows += month_rows
        months_completed += 1

        log(f"  [COMPLETED] {month_key}: {month_rows:,} rows successfully written to disk")
        time.sleep(2)

    db.close()
    log(f"\nCOLLECTION COMPLETE | Rows processed: {total_rows:,}")

if __name__ == "__main__":
    run_collection()