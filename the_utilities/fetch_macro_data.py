import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

from the_utilities.paths import RAW_MACRO_CSV


def update_macro_data():
    # Appends new daily macro data to the existing CSV
    # Only downloads days that are missing from the current file

    tickers = ["SPY", "^VIX", "IEF", "HYG"]
    os.makedirs(os.path.dirname(RAW_MACRO_CSV), exist_ok=True)

    # 1. Load existing data to find the last date
    if os.path.exists(RAW_MACRO_CSV):
        existing = pd.read_csv(RAW_MACRO_CSV, index_col='Date', parse_dates=True)
        last_date = existing.index.max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"[SYSTEM] Existing data through {last_date.date()}. Fetching from {start_date}...")
    else:
        existing = pd.DataFrame()
        start_date = "2010-01-01"
        print(f"[SYSTEM] No existing file found. Full download from {start_date}...")

    # 2. Download only the missing days
    end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date >= end_date:
        print("[INFO] Macro data already up to date. Nothing to fetch.")
        return existing

    new_data = yf.download(tickers, start=start_date, end=end_date)['Close']

    if new_data.empty:
        print("[INFO] No new data available from Yahoo Finance.")
        return existing

    new_data = new_data.ffill().dropna()

    # 3. Append to existing and deduplicate
    if not existing.empty:
        combined = pd.concat([existing, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
    else:
        combined = new_data

    # 4. Atomic write
    temp_path = RAW_MACRO_CSV + ".tmp"
    combined.to_csv(temp_path)
    os.replace(temp_path, RAW_MACRO_CSV)

    new_rows = len(combined) - len(existing)
    print(f"[SUCCESS] Appended {new_rows} new days. Total: {len(combined)} rows.")
    return combined


def rebuild_macro_data():
    # Full re-download from scratch — use only when the CSV is corrupted or missing
    tickers = ["SPY", "^VIX", "IEF", "HYG"]
    print(f"[SYSTEM] Full rebuild: downloading {tickers} from 2010 to present...")

    data = yf.download(tickers, start="2010-01-01", end=datetime.now().strftime('%Y-%m-%d'))['Close']
    data = data.ffill().dropna()

    os.makedirs(os.path.dirname(RAW_MACRO_CSV), exist_ok=True)
    temp_path = RAW_MACRO_CSV + ".tmp"
    data.to_csv(temp_path)
    os.replace(temp_path, RAW_MACRO_CSV)

    print(f"[SUCCESS] Rebuilt macro data: {len(data)} rows saved to {RAW_MACRO_CSV}")
    return data


if __name__ == "__main__":
    print("\n====== MACRO DATA UPDATER ======")
    df = update_macro_data()
    print(f"\n[DONE] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(df.tail(3))