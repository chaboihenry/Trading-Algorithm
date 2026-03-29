import os
import time
import json
import threading
import pandas as pd
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

# Load credentials
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

# Global RAM buffer and paths
tick_buffer = {}
UNIVERSE_PATH = "the_models/curated_universe.json"
last_modified = 0.0

def load_curated_universe(limit=30):
    # Load and validate the universe file
    if not os.path.exists(UNIVERSE_PATH):
        print(f"[WARNING] {UNIVERSE_PATH} missing.")
        return []

    with open(UNIVERSE_PATH, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    universe = data.get("symbols", []) if isinstance(data, dict) else data
    
    if not isinstance(universe, list):
        print("[ERROR] JSON format invalid.")
        return []
        
    return universe[:limit]

def get_file_timestamp(filepath):
    # Check last modified time
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0

def flush_to_parquet(ticker):
    # Save buffered ticks to ZSTD compressed parquet
    if ticker not in tick_buffer or not tick_buffer[ticker]: 
        return

    base_path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
    os.makedirs(base_path, exist_ok=True)

    df = pd.DataFrame(tick_buffer[ticker])
    file_path = os.path.join(base_path, f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}.parquet")

    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path, compression='zstd')

    tick_buffer[ticker] = []
    print(f"[Scraper] Flushed {len(df)} ticks for {ticker}.")

async def handle_trade(trade):
    # Process incoming trades and append to buffer
    ticker = trade.symbol
    if ticker not in tick_buffer:
        tick_buffer[ticker] = []

    tick_buffer[ticker].append({
        'timestamp': trade.timestamp,
        'price': trade.price,
        'size': trade.size,
        'exchange': trade.exchange
    })

    # Flush once buffer hits 10k ticks
    if len(tick_buffer[ticker]) >= 10000:
        flush_to_parquet(ticker)

def universe_watcher(stream_instance):
    # Background thread to monitor file changes and kill stream
    global last_modified
    while True:
        time.sleep(10) # Poll every 10 seconds
        current_ts = get_file_timestamp(UNIVERSE_PATH)
        
        if current_ts > last_modified:
            print("\n[UPDATE] Change detected in curated_universe.json.")
            last_modified = current_ts
            # Calling stop() allows the main loop to progress
            stream_instance.stop()
            break 

if __name__ == "__main__":
    print("====== Initializing M1 Hot-Reload Scraper ======")
    
    retry_delay = 5
    last_modified = get_file_timestamp(UNIVERSE_PATH)

    while True:
        try:
            # 1. Load the symbols
            UNIVERSE = load_curated_universe(limit=30)
            if not UNIVERSE:
                time.sleep(30)
                continue

            # 2. Prep buffers
            for t in UNIVERSE:
                if t not in tick_buffer: 
                    tick_buffer[t] = []

            # 3. Setup Stream
            stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)
            stream.subscribe_trades(handle_trade, *UNIVERSE)
            
            # 4. Launch file watcher thread
            watcher = threading.Thread(target=universe_watcher, args=(stream,), daemon=True)
            watcher.start()

            print(f"Connected. Monitoring {len(UNIVERSE)} curated assets...")
            retry_delay = 5
            
            # 5. Execute blocking stream
            stream.run()

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Manual override. Flushing data...")
            for ticker in list(tick_buffer.keys()):
                flush_to_parquet(ticker)
            break
            
        except Exception as e:
            print(f"\n[STREAM ERROR] {e}")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)