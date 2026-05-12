import os
import json
import asyncio
import logging
import threading
import pandas as pd
from datetime import datetime
from alpaca_trade_api.stream import Stream

from the_utilities.paths import CURATED_UNIVERSE_JSON

logger = logging.getLogger(__name__)


class LiveStreamer:
    # Asynchronous WebSocket streamer.
    # Maintains a rolling live matrix of prices for the active curated universe.
    def __init__(self, api_key: str, secret_key: str, base_url: str, logger=None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        self.live_matrix = pd.DataFrame()
        self._matrix_lock = threading.Lock()
        self.active_tickers = []

        self._load_universe()

        # Initialize the Alpaca Stream with Free Tier compliance
        self.stream = Stream(
            self.api_key,
            self.secret_key,
            base_url=self.base_url,
            data_feed='iex'
        )

    def _load_universe(self):
        # Extracts the flat_list of tickers from the M1's JSON payload.
        with open(CURATED_UNIVERSE_JSON, "r") as f:
            data = json.load(f)
            self.active_tickers = data.get("flat_list", [])
            
        self.logger.info(f"[SUCCESS] Loaded {len(self.active_tickers)} active tickers from payload.")
        
        # Initialize empty columns for the matrix
        self.live_matrix = pd.DataFrame(columns=self.active_tickers)

    async def _handle_bar(self, bar):
        ticker = bar.symbol
        price = bar.close
        timestamp = bar.timestamp
    
        with self._matrix_lock:
            if timestamp not in self.live_matrix.index:
                self.live_matrix.loc[timestamp] = None
            self.live_matrix.at[timestamp, ticker] = price
            self.live_matrix = self.live_matrix.ffill()
            if len(self.live_matrix) > 200:
                self.live_matrix = self.live_matrix.tail(200)

    def get_latest_matrix(self) -> pd.DataFrame:
        with self._matrix_lock:
            return self.live_matrix.dropna().copy()

    def start_streaming(self):
        # Subscribes to the tickers and starts the infinite async loop.
        if not self.active_tickers:
            self.logger.warning("[WARNING] No tickers to stream. Exiting.")
            return

        self.logger.info(f"[SYSTEM] Connecting to Alpaca IEX WebSocket for {len(self.active_tickers)} assets...")
        
        # Subscribe to minute bars for all active tickers
        self.stream.subscribe_bars(self._handle_bar, *self.active_tickers)
        
        # Start the stream (this blocks the thread it runs on)
        self.stream.run()

if __name__ == "__main__":
    from dotenv import load_dotenv
    import threading
    import time

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("====== EXECUTION NODE DIAGNOSTIC: LIVE STREAMER ======")

    # 1. Load environment variables for the diagnostic run
    load_dotenv()
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_API_SECRET")
    BASE_URL = os.getenv("ALPACA_API_BASE_URL", "https://paper-api.alpaca.markets/v2")

    if not API_KEY or not SECRET_KEY:
        logger.error("[CRITICAL] Alpaca API keys not found in .env file.")
        exit(1)

    logger.info("[SYSTEM] Environment variables loaded. Initializing Streamer...")

    # 2. Initialize Streamer (Tests the Fail-Fast JSON logic)
    streamer = LiveStreamer(
        api_key=API_KEY,
        secret_key=SECRET_KEY,
        base_url=BASE_URL,
    )

    # 3. Run the stream in a background daemon thread
    logger.info("[SYSTEM] Initiating 15-second WebSocket diagnostic test...")
    stream_thread = threading.Thread(target=streamer.start_streaming, daemon=True)
    stream_thread.start()

    # Wait for the WebSocket to connect and gather ticks
    time.sleep(15)

    # 4. Evaluate the Live Matrix
    logger.info("====== DIAGNOSTIC RESULTS ======")
    matrix = streamer.get_latest_matrix()

    if matrix.empty:
        logger.warning(
            "[WARNING] Matrix is empty. This is expected if the market is closed "
            "or IEX volume is zero during this 15-second window."
        )
    else:
        logger.info(f"[SUCCESS] Live Matrix Shape: {matrix.shape}")
        logger.info(f"Latest DataFrame Rows:\n{matrix.tail(3)}")

    # 5. Clean teardown
    logger.info("[SYSTEM] Diagnostic complete. Terminating WebSocket.")
    streamer.stream.stop()