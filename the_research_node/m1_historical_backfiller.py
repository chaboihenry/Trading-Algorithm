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

# Define the exact same strict schema as the scraper to prevent ArrowInvalid crashes
STRICT_SCHEMA = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    ('price', pa.float64()),
    ('size', pa.float64()),
    ('exchange', pa.string())
])

def load_universe():
    with open('universe.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_vault_bounds(ticker: str):
    # Finds the true mathematical boundaries of the data, immune to sorting errors
    path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
    if not os.path.exists(path) or not os.listdir(path):
        return None
    
    # 1. Filter out Apple '._' ghost files natively so the Janitor ignores them
    files = [f for f in os.listdir(path) if f.endswith('.parquet') and not f.startswith('._')]
    if not files: 
        return None

    global_start = None
    global_end = None

    # 2. Scan ALL valid files to find the true min and max dates
    # This bypasses the alphabetical sorting bug between "HIST_" and "2026..."
    for f in files:
        file_path = os.path.join(path, f)
        try:
            # Reading ONLY the timestamp column is highly optimized in PyArrow
            df = pd.read_parquet(file_path, columns=['timestamp'])
            if df.empty: continue
            
            file_min = df['timestamp'].min().replace(tzinfo=None)
            file_max = df['timestamp'].max().replace(tzinfo=None)
            
            if global_start is None or file_min < global_start:
                global_start = file_min
            if global_end is None or file_max > global_end:
                global_end = file_max
                
        except Exception as e:
            # If PyArrow actually crashes on a real file, delete it
            print(f"  >> [JANITOR] Removing corrupted file: {f}")
            os.remove(file_path)

    if global_start and global_end:
        return global_start, global_end
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
                    
                    # Standardize timestamp and explicitly cast columns to match STRICT_SCHEMA
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df['price'] = df['price'].astype(float)
                    df['size'] = df['size'].astype(float)
                    df['exchange'] = df['exchange'].astype(str)
                    
                    base_path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
                    os.makedirs(base_path, exist_ok=True)
                    filename = f"HIST_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.parquet"
                    file_path = os.path.join(base_path, filename)
                    
                    # Apply the STRICT_SCHEMA here to prevent ArrowInvalid schema unification crashes
                    table = pa.Table.from_pandas(df[['timestamp', 'price', 'size', 'exchange']], schema=STRICT_SCHEMA)
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