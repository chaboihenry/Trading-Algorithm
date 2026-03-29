import os
import time
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

# Global RAM buffer
tick_buffer = {}

def load_universe():
    with open('universe.txt', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def flush_to_parquet(ticker: str):
    if ticker not in tick_buffer or not tick_buffer[ticker]: 
        return

    base_path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
    os.makedirs(base_path, exist_ok=True)

    df = pd.DataFrame(tick_buffer[ticker])
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}.parquet"
    file_path = os.path.join(base_path, filename)

    # High compression ZSTD for the 2TB SSD
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path, compression='zstd')

    tick_buffer[ticker] = []
    print(f"[Scraper] Flushed {len(df)} ticks for {ticker} to SSD.")

async def handle_trade(trade):
    ticker = trade.symbol
    if ticker not in tick_buffer:
        tick_buffer[ticker] = []

    tick_buffer[ticker].append({
        'timestamp': trade.timestamp,
        'price': trade.price,
        'size': trade.size,
        'exchange': trade.exchange
    })

    # High-volume threshold (Change 10000 back to 10 for small tests)
    if len(tick_buffer[ticker]) >= 10000:
        flush_to_parquet(ticker)

if __name__ == "__main__":
    print("====== Initializing M1 Robust Async Scraper ======")
    
    # Connection parameters
    retry_delay = 5  # Start with 5 seconds
    max_retry = 60   # Max wait of 1 minute

    while True:
        try:
            # 1. Reload universe in case you added stocks while it was down
            UNIVERSE = load_universe()
            for t in UNIVERSE:
                if t not in tick_buffer: tick_buffer[t] = []

            # 2. Initialize fresh stream
            stream = StockDataStream(API_KEY, SECRET_KEY, feed=DataFeed.IEX)
            
            # 3. Subscribe and Run
            stream.subscribe_trades(handle_trade, *UNIVERSE)
            print(f"Connected. Monitoring {len(UNIVERSE)} assets...")
            
            # Reset retry delay on successful connection
            retry_delay = 5
            stream.run()

        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Manual override. Flushing remaining data...")
            for ticker in tick_buffer.keys():
                flush_to_parquet(ticker)
            print("Safe shutdown complete.")
            break
        
        except Exception as e:
            print(f"[CONNECTION ERROR] {e}")
            print(f"Attempting reconnection in {retry_delay} seconds...")
            time.sleep(retry_delay)
            
            # Exponential backoff so we don't spam Alpaca and get banned
            retry_delay = min(retry_delay * 2, max_retry)