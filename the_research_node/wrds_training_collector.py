import os
import gc
import json
import subprocess
import time
import wrds
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

from the_utilities.paths import LOGS_DIR, VAULT_ROOT

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

# Share-class encoded tickers: TAQ stores these as (sym_root, sym_suffix)
# rather than the plain ticker. Add entries when universe_sweep finds them.
TICKER_MAPPING = {
    "GOOGL": ("GOOG", "L"),  # Alphabet Class A
    "CMCSA": ("CMCS", "A"),  # Comcast Class A
    "BRK-B": ("BRK", "B"),   # Berkshire Hathaway Class B
    "BF-B":  ("BF",  "B"),   # Brown-Forman Class B
}

# Error substrings that indicate the WRDS connection has died and needs reconnect
CONNECTION_ERROR_PATTERNS = [
    "Can't reconnect until invalid transaction",
    "Connection timed out",
    "SSL SYSCALL error",
    "server closed the connection",
    "could not receive data from server",
    "InvalidatedTransaction",
]


class _ConnectionBroken(Exception):
    """Internal signal: WRDS connection died, caller should reconnect."""
    pass


def ensure_vault_accessible(max_retries: int = 5) -> bool:
    """
    Check that VAULT_ROOT is accessible. If not, attempt to remount the SSD
    via drvfs and retry. Returns True if vault is accessible after attempts,
    False otherwise.
    """
    for attempt in range(max_retries):
        # First, check if path is currently accessible
        try:
            if os.path.exists(VAULT_ROOT):
                # Touch a probe to confirm it's not stale
                probe = os.path.join(VAULT_ROOT, '.mount_probe')
                try:
                    with open(probe, 'w') as f:
                        f.write('ok')
                    os.remove(probe)
                    return True
                except OSError:
                    pass  # Stale mount — proceed to remount
        except OSError:
            pass

        # Attempt remount
        print(f'[REMOUNT] Vault inaccessible (attempt {attempt+1}/{max_retries}), remounting...')
        subprocess.run(['sudo', '-n', 'umount', '-l', '/mnt/e'],
                       stderr=subprocess.DEVNULL)
        subprocess.run(['sudo', '-n', 'mount', '-t', 'drvfs', 'E:', '/mnt/e',
                        '-o', 'uid=1000,gid=1000'],
                       stderr=subprocess.DEVNULL)
        time.sleep(2)

    return False


def get_tickers_needing_data(universe: list, month_key: str) -> list:
    """
    Return the subset of tickers from universe that don't already have a parquet
    file for the given month. Used to filter the WRDS query to ONLY tickers we
    actually need data for, so universe expansions don't trigger redundant
    re-queries of already-pulled tickers.

    Args:
        universe: list of ticker symbols (user-facing, e.g. 'BRK-B', 'GOOGL')
        month_key: month identifier in 'YYYY_MM' format

    Returns:
        list of tickers from universe whose parquet for this month is missing
    """
    # Per-ticker file-existence check — gate is the actual parquet file
    needed = []
    for ticker in universe:
        parquet_path = os.path.join(
            VAULT_ROOT, ticker, 'parquet', 'training_data',
            f'{month_key}.parquet'
        )
        if not os.path.exists(parquet_path):
            needed.append(ticker)
    return needed


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
    # Scan vault to find the most recent month file across all tickers
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


def _is_connection_broken(error_msg):
    # True if the error message contains any known connection-failure pattern
    return any(p in error_msg for p in CONNECTION_ERROR_PATTERNS)


def _reconnect(db):
    # Close broken connection (silently) and open a fresh one
    try:
        db.close()
    except Exception:
        pass
    log("  [RECOVERY] Opening new WRDS connection...")
    new_db = wrds.Connection(wrds_username=WRDS_USERNAME)
    log("  [RECOVERY] Reconnected.")
    return new_db


def query_single_day(db, date_str, ticker_batch):
    # Library naming: taqmsec is deprecated for new data; use taqm_YYYY
    year = date_str[:4]
    library = f"taqm_{year}"
    table_name = f"ctm_{date_str}"

    # Split batch into normal tickers and share-class-encoded ones
    normal = [t for t in ticker_batch if t not in TICKER_MAPPING]
    special = [t for t in ticker_batch if t in TICKER_MAPPING]

    # Normal tickers: sym_root match + NULL suffix
    # Special tickers: explicit (root, suffix) match
    where_parts = []
    if normal:
        normal_sql = ", ".join(f"'{t}'" for t in normal)
        where_parts.append(f"(sym_root IN ({normal_sql}) AND sym_suffix IS NULL)")
    for t in special:
        root, suffix = TICKER_MAPPING[t]
        where_parts.append(f"(sym_root = '{root}' AND sym_suffix = '{suffix}')")
    where_sql = " OR ".join(where_parts)

    # Pull sym_suffix so we can remap share-class rows back to user-facing names
    query = f"""
        SELECT date, time_m, price, size, ex, sym_root, sym_suffix
        FROM {library}.{table_name}
        WHERE ({where_sql})
        AND tr_corr = '00'
        AND price > 0
    """
    try:
        df = db.raw_sql(query)
        # Remap special rows so sym_root carries the user-facing ticker
        if "sym_suffix" in df.columns:
            if not df.empty:
                for ticker, (root, suffix) in TICKER_MAPPING.items():
                    mask = (df["sym_root"] == root) & (df["sym_suffix"] == suffix)
                    df.loc[mask, "sym_root"] = ticker
            df = df.drop(columns=["sym_suffix"])
        return df
    except Exception as e:
        error_msg = str(e)
        if "does not exist" in error_msg or "UndefinedTable" in error_msg:
            # Holidays and non-trading days — silent skip is intentional
            return pd.DataFrame()
        if _is_connection_broken(error_msg):
            # Signal caller to reconnect rather than swallowing this silently
            raise _ConnectionBroken(error_msg[:200])
        log(f"  [ERROR] Query failed for {table_name}: {error_msg}")
        return pd.DataFrame()


def query_with_recovery(db, date_str, ticker_batch, max_attempts=3):
    # Wrap query_single_day with auto-reconnect on connection failures.
    # Returns (df, possibly_new_db).
    for attempt in range(max_attempts):
        try:
            df = query_single_day(db, date_str, ticker_batch)
            return df, db
        except _ConnectionBroken as e:
            log(f"  [RECOVERY] Connection lost ({attempt+1}/{max_attempts}): {str(e)[:120]}")
            if attempt + 1 < max_attempts:
                time.sleep(15 * (attempt + 1))  # backoff: 15s, 30s
                db = _reconnect(db)
    log(f"  [GIVE UP] Could not recover for {date_str}, skipping this batch")
    return pd.DataFrame(), db


def clean_daily_batch(df):
    # Clean and deduplicate the dataframe immediately to free RAM
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
    # Detect the last collected month and fetch only new data
    log("====== WRDS TAQ INCREMENTAL COLLECTOR ======")

    universe = load_universe()
    log(f"Universe: {len(universe)} tickers")

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
                df, db = query_with_recovery(db, date_str, batch)

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
    log(f"\n[DONE] Incremental collection complete. {total_rows:,} new rows processed.")

def run_full_rebuild(start_year: int = 2021):
    # Full collection from scratch — use when vault is empty or corrupted

    # Verify SSD vault is accessible before starting
    if not ensure_vault_accessible():
        raise RuntimeError('Cannot access vault at startup. Check SSD connection.')

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

        # Per-ticker skip: only query tickers whose parquet for this month is missing.
        # Universe expansions no longer trigger redundant re-queries of pulled tickers.
        tickers_needed = get_tickers_needing_data(universe, month_key)
        if not tickers_needed:
            log(f"  [SKIP] {month_key}: all {len(universe)} tickers already have data")
            continue

        log(f"  [PROCESSING] {month_key}: querying {len(tickers_needed)}/{len(universe)} tickers")

        marker_path = os.path.join(VAULT_ROOT, "_markers", f"{month_key}.done")

        writers = {}
        for ticker in tickers_needed:
            ticker_dir = os.path.join(VAULT_ROOT, ticker, "parquet", "training_data")
            # Retry per-ticker writer setup on SSD disconnect (Errno 19)
            write_attempts = 0
            while write_attempts < 3:
                try:
                    os.makedirs(ticker_dir, exist_ok=True)
                    file_path = os.path.join(ticker_dir, f"{month_key}.parquet")
                    writers[ticker] = pq.ParquetWriter(file_path, STRICT_SCHEMA, compression="zstd")
                    break  # Success, exit retry loop
                except OSError as e:
                    if e.errno == 19:  # No such device
                        print(f'[ERROR] SSD disconnected during write for {ticker}, attempting remount...')
                        if not ensure_vault_accessible():
                            raise RuntimeError('Vault unrecoverable after remount attempts')
                        write_attempts += 1
                        time.sleep(5)
                    else:
                        raise  # Different error, propagate

        month_rows = 0

        for day_idx, day in enumerate(days):
            date_str = day.strftime("%Y%m%d")

            for batch_start in range(0, len(tickers_needed), TICKER_BATCH_SIZE):
                batch = tickers_needed[batch_start: batch_start + TICKER_BATCH_SIZE]
                df, db = query_with_recovery(db, date_str, batch)

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

        for ticker in tickers_needed:
            writers[ticker].close()

        # Marker file: advisory only — records which months were attempted by which collector
        # run. Skip-logic is now per-ticker file existence (see get_tickers_needing_data).
        if month_rows > 0:
            os.makedirs(os.path.dirname(marker_path), exist_ok=True)
            with open(marker_path, "w") as f:
                f.write(f"{datetime.now().isoformat()}\n{month_rows} rows\n")
            log(f"  [COMPLETED] {month_key}: {month_rows:,} rows written to disk")
        else:
            log(f"  [WARN] {month_key}: 0 rows; marker NOT written")

        total_rows += month_rows
        time.sleep(2)

    db.close()
    log(f"\n[DONE] Full rebuild complete. {total_rows:,} rows processed.")


if __name__ == "__main__":
    # Full 5-year rebuild on ASUS WSL2 — start from Jan 2021
    run_full_rebuild(start_year=2021)