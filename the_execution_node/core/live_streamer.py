import os
import json
import asyncio
import pandas as pd
from datetime import datetime
from alpaca_trade_api.stream import Stream

class LiveStreamer:
    # Asynchronous WebSocket streamer.
    #Maintains a rolling live matrix of prices for the active curated universe.
    def __init__(self, api_key: str, secret_key: str, base_url: str, models_dir: str = "the_models"):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.models_dir = models_dir
        
        # 1. Initialize the rolling window for live prices
        self.live_matrix = pd.DataFrame()
        self.active_tickers = []
        
        # 2. Load the universe immediately upon initialization (Will crash if missing)
        self._load_universe()
        
        # 3. Initialize the Alpaca Stream with Free Tier compliance
        self.stream = Stream(
            self.api_key,
            self.secret_key,
            base_url=self.base_url,
            data_feed='iex'
        )

    def _load_universe(self):
        # Extracts the flat_list of tickers from the M1's JSON payload.
        path = os.path.join(self.models_dir, "curated_universe.json")
        
        with open(path, "r") as f:
            data = json.load(f)
            self.active_tickers = data.get("flat_list", [])
            
        print(f"[SUCCESS] Loaded {len(self.active_tickers)} active tickers from payload.")
        
        # Initialize empty columns for the matrix
        self.live_matrix = pd.DataFrame(columns=self.active_tickers)

    async def _handle_bar(self, bar):
        # Callback function executed every time a new 1-minute bar arrives
        ticker = bar.symbol
        price = bar.close
        timestamp = bar.timestamp
        
        # 1. Update the live matrix with the new timestamp if it doesn't exist
        if timestamp not in self.live_matrix.index:
            self.live_matrix.loc[timestamp] = None
            
        self.live_matrix.at[timestamp, ticker] = price
        
        # 2. Forward fill to handle missing ticks across the basket, then drop NAs
        self.live_matrix = self.live_matrix.ffill()
        
        # 3. Memory Management: Keep only the last 200 bars (intraday rolling window)
        if len(self.live_matrix) > 200:
            self.live_matrix = self.live_matrix.tail(200)

    def get_latest_matrix(self) -> pd.DataFrame:
        # Returns a clean, safely copied snapshot of the current live matrix
        # Drop rows that don't have a complete set of prices yet
        return self.live_matrix.dropna().copy()

    def start_streaming(self):
        # Subscribes to the tickers and starts the infinite async loop.
        if not self.active_tickers:
            print("[WARNING] No tickers to stream. Exiting.")
            return

        print(f"[SYSTEM] Connecting to Alpaca IEX WebSocket for {len(self.active_tickers)} assets...")
        
        # Subscribe to minute bars for all active tickers
        self.stream.subscribe_bars(self._handle_bar, *self.active_tickers)
        
        # Start the stream (this blocks the thread it runs on)
        self.stream.run()

if __name__ == "__main__":
    from dotenv import load_dotenv
    import threading
    import time

    print("\n====== EXECUTION NODE DIAGNOSTIC: LIVE STREAMER ======")

    # 1. Load environment variables for the diagnostic run
    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_API_SECRET")
    BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not API_KEY or not SECRET_KEY:
        print("[CRITICAL] Alpaca API keys not found in .env file.")
        exit(1)

    print("[SYSTEM] Environment variables loaded. Initializing Streamer...")

    # 2. Initialize Streamer (Tests the Fail-Fast JSON logic)
    streamer = LiveStreamer(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        base_url=BASE_URL,
        models_dir="the_models"
    )

    # 3. Run the stream in a background daemon thread
    print("\n[SYSTEM] Initiating 15-second WebSocket diagnostic test...")
    stream_thread = threading.Thread(target=streamer.start_streaming, daemon=True)
    stream_thread.start()

    # Wait for the WebSocket to connect and gather ticks
    time.sleep(15)

    # 4. Evaluate the Live Matrix
    print("\n====== DIAGNOSTIC RESULTS ======")
    matrix = streamer.get_latest_matrix()
    
    if matrix.empty:
        print("[WARNING] Matrix is empty. This is expected if the market is closed or IEX volume is zero during this 15-second window.")
    else:
        print(f"[SUCCESS] Live Matrix Shape: {matrix.shape}")
        print("\nLatest DataFrame Rows:")
        print(matrix.tail(3))

    # 5. Clean teardown
    print("\n[SYSTEM] Diagnostic complete. Terminating WebSocket.")
    streamer.stream.stop()