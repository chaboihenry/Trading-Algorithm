import os
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

class MarketOracle:
    def __init__(self):
        # securely load API keys
        load_dotenv()
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
            raise ValueError("CRITICAL: Alpaca keys not found in environment.")

        # initialize the Historical Data Client
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        print("[The Oracle] Neural link to Alpaca established.")

    def get_live_5m_bars(self, tickers: list, lookback_days: int = 7):
        # Fetches the rolling 7-day window of 5-minute bars and standardizes 
        # it into a single matrix for the brain to process in the RAM.

        # Calculate the rolling time window
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)
        
        print(f"[The Oracle] Requesting 5-minute tape for {len(tickers)} assets...")
        
        # configure the exact API request
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX 
        )
        
        try:
            # fetch the data into memory
            bars = self.client.get_stock_bars(request_params).df
        except Exception as e:
            print(f"[ERROR] The Oracle failed to fetch data: {e}")
            return pd.DataFrame()
        
        if bars.empty:
            print("[WARNING] Alpaca returned empty data. Is the market closed?")
            return pd.DataFrame()

        # Alpaca returns a MultiIndex (symbol, timestamp). 
        # pivot it so the index is the Timestamp, and columns are Tickers with their Close prices.
        df_close = bars.reset_index().pivot(index='timestamp', columns='symbol', values='close')
        
        # forward-fill any missing bars (e.g., due to low liquidity/halts), then drop leading NaNs
        df_close = df_close.ffill().dropna()
        
        return df_close

if __name__ == "__main__":
    print("=== TESTING THE ORACLE ===")
    
    # standard mega-cap stat-arb universe
    test_universe = ['V', 'MA', 'AAPL', 'MSFT', 'NVDA', 'AMD', 'JPM', 'BAC']
    
    oracle = MarketOracle()
    live_matrix = oracle.get_live_5m_bars(tickers=test_universe, lookback_days=5)
    
    if not live_matrix.empty:
        print("\nSuccess: Matrix Standardized. Shape:", live_matrix.shape)
        print("\nTail of the Live Matrix:")
        print(live_matrix.tail())