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
FLUSH_THRESHOLD = 10000  # Flush memory to SSD every 10,000 ticks per ticker

def load_curated_universe(limit=30):
    # Strict loading logic based on the new JSON architecture
    if not os.path.exists(UNIVERSE_PATH):
        return []

    with open(UNIVERSE_PATH, 'r') as f:
        data = json.load(f)
    
    # Target the correct key from our new structured payload
    universe = data.get("flat_list", [])
        
    return universe[:limit]

def get_file_timestamp(filepath):
    return os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0

def flush_to_parquet(ticker):
    # Compress and flush the tick buffer to SSD using a strict schema
    if ticker not in tick_buffer or not tick_buffer[ticker]: 
        return

    base_path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"
    os.makedirs(base_path, exist_ok=True)

    df = pd.DataFrame(tick_buffer[ticker])
    
    # Standardize the pandas timestamp first
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # Explicitly define the unbreakable PyArrow schema
    strict_schema = pa.schema([
        ('timestamp', pa.timestamp('ns', tz='UTC')),
        ('price', pa.float64()),
        ('size', pa.float64()),
        ('exchange', pa.string())
    ])

    file_path = os.path.join(base_path, f"{pd.Timestamp.now().strftime('%Y%m%d%H%M%S%f')}.parquet")

    # Force the dataframe into the strict schema during Table creation
    table = pa.Table.from_pandas(df, schema=strict_schema)
    pq.write_table(table, file_path, compression='zstd')

    # Clear the RAM buffer
    tick_buffer[ticker] = []
    print(f"[Scraper] Flushed {len(df)} ticks for {ticker} (Strict 'ns' Schema).")

async def handle_trade(trade):
    # Asynchronous callback to process incoming Alpaca ticks
    ticker = trade.symbol
    if ticker not in tick_buffer:
        tick_buffer[ticker] = []
        
    tick_buffer[ticker].append({
        'timestamp': trade.timestamp,
        'price': float(trade.price),
        'size': float(trade.size),
        'exchange': str(trade.exchange)
    })
    
    # RAM Management: Flush to the Vault if the buffer hits the threshold
    if len(tick_buffer[ticker]) >= FLUSH_THRESHOLD:
        flush_to_parquet(ticker)

def universe_watcher(stream_instance):
    # Background thread to monitor the atomic writes from the discovery script
    global last_modified
    while True:
        time.sleep(10) # Poll every 10 seconds
        current_ts = get_file_timestamp(UNIVERSE_PATH)
        
        if current_ts > last_modified:
            print("\n[UPDATE] Change detected in curated_universe.json.")
            last_modified = current_ts
            # Calling stop() gracefully breaks the stream.run() blocking loop
            stream_instance.stop()
            break 

if __name__ == "__main__":
    print("====== Initializing M1 Hot-Reload Scraper ======")
    
    last_modified = get_file_timestamp(UNIVERSE_PATH)

    while True:
        # 1. Load the symbols directly from the structured JSON
        UNIVERSE = load_curated_universe(limit=30)
        
        # If the file is missing or empty, wait and retry. 
        if not UNIVERSE:
            print("[WAITING] Universe list empty or not found. Retrying in 30s...")
            time.sleep(30)
            continue

        # 2. Prep buffers
        for t in UNIVERSE:
            if t not in tick_buffer: 
                tick_buffer[t] = []

        # 3. Setup Stream
        stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)
        stream.subscribe_trades(handle_trade, *UNIVERSE)
        
        # 4. Launch atomic file watcher thread
        watcher = threading.Thread(target=universe_watcher, args=(stream,), daemon=True)
        watcher.start()

        print(f"Connected. Monitoring {len(UNIVERSE)} curated assets...")
        
        # 5. Execute blocking stream
        # If Alpaca drops the websocket connection, this will throw a hard error and crash.
        # A process manager will restart it fresh.
        stream.run()
        
        # 6. Graceful flush if stopped by the universe_watcher
        print("\n[RELOAD] Stream stopped by watcher. Flushing data before reload...")
        for ticker in list(tick_buffer.keys()):
            flush_to_parquet(ticker)