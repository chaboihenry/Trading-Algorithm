import os
import pandas as pd
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockTradesRequest

# load Alpaca credentials from the .env file
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    print("[CRITICAL] The .env file was not loaded correctly. Keys are missing.")
    exit()

# initialize the historical client
client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

def load_universe():
    with open('universe.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_vault_bounds(ticker: str):
    """Finds the date boundaries and automatically deletes corrupted files."""
    path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
    if not os.path.exists(path) or not os.listdir(path):
        return None
    
    files = sorted([f for f in os.listdir(path) if f.endswith('.parquet')])
    if not files: 
        return None

    start_date = None
    end_date = None

    # 1. Scan forward to find the true Start Date
    for f in files:
        file_path = os.path.join(path, f)
        try:
            df = pd.read_parquet(file_path, columns=['timestamp'])
            start_date = df['timestamp'].min().replace(tzinfo=None)
            break  # Found the first valid file, stop looking
        except Exception:
            print(f"  >> [JANITOR] Deleting corrupted file: {f}")
            os.remove(file_path)

    # Refresh the file list in case the janitor deleted files
    files = sorted([f for f in os.listdir(path) if f.endswith('.parquet')])
    if not files or not start_date: 
        return None

    # 2. Scan backward to find the true End Date
    for f in reversed(files):
        file_path = os.path.join(path, f)
        try:
            df = pd.read_parquet(file_path, columns=['timestamp'])
            end_date = df['timestamp'].max().replace(tzinfo=None)
            break  # Found the last valid file, stop looking
        except Exception:
            print(f"  >> [JANITOR] Deleting corrupted file: {f}")
            os.remove(file_path)

    if start_date and end_date:
        return start_date, end_date
    return None

def backfill_ticker(ticker: str, target_start: datetime):
    print(f"\n[Backfill] Analyzing {ticker}...")
    bounds = get_vault_bounds(ticker)
    
    # Store the ranges needed to fetch in a list
    fetch_ranges = []
    
    if not bounds:
        # Scenario 1: The Vault is completely empty. Fetch everything.
        fetch_ranges.append((target_start, datetime.now()))
    else:
        existing_start, existing_end = bounds
        
        # Scenario 2: The "Past" Gap (2023 -> Start of prev data)
        if existing_start > target_start:
            print(f"  >> Past Gap detected: {target_start.date()} to {existing_start.date()}")
            fetch_ranges.append((target_start, existing_start))
            
        # Scenario 3: The "Recent" Gap (End of prev data -> Today)
        # add a 1-day buffer to avoid redundant overlapping fetches
        if existing_end < (datetime.now() - timedelta(days=1)):
            print(f"  >> Recent Gap detected: {existing_end.date()} to Today")
            fetch_ranges.append((existing_end, datetime.now()))

    if not fetch_ranges:
        print(f"  >> Vault is fully up to date for {ticker}. Skipping.")
        return

    # Execute the fetches for whatever gaps found
    for start_date, end_date in fetch_ranges:
        current_start = start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=30), end_date)
            print(f"  >> Fetching chunk: {current_start.date()} to {current_end.date()}...")
            
            request_params = StockTradesRequest(
                symbol_or_symbols=ticker,
                start=current_start,
                end=current_end,
                feed="iex"
            )
            
            try:
                trades = client.get_stock_trades(request_params)
                if trades and not trades.df.empty:
                    df = trades.df.reset_index().rename(columns={'symbol': 'ticker'})
                    
                    base_path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
                    os.makedirs(base_path, exist_ok=True)
                    filename = f"HIST_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.parquet"
                    file_path = os.path.join(base_path, filename)
                    
                    table = pa.Table.from_pandas(df[['timestamp', 'price', 'size', 'exchange']])
                    pq.write_table(table, file_path, compression='zstd')
                    print(f"  >> [SUCCESS] Flushed chunk to Vault.")
                else:
                    print("  >> [WARNING] No data returned for this chunk.")
            except Exception as e:
                print(f"  >> [ERROR] API fetch failed: {e}")
            
            current_start = current_end

if __name__ == "__main__":
    print("====== Initializing M1 Historical Backfiller ======")
    
    UNIVERSE = load_universe()
    print(f"Loaded {len(UNIVERSE)} assets from universe.txt.")
    
    # target exactly 3 years back from today
    TARGET_DATE = datetime(2023, 3, 28) 
    
    for ticker in UNIVERSE:
        backfill_ticker(ticker, TARGET_DATE)
        
    print("\n====== Backfill Complete ======")