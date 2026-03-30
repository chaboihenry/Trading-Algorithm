import os
import pandas as pd
import yfinance as yf

def fetch_macro_data(start_date: str, end_date: str, save_path: str = 'the_execution_node/data/raw_macro_data.csv'):
    # Fetches daily adjusted close prices for macroeconomic proxies.
    # Aligns dates and handles missing values by forward-filling.
    
    # 1. Define target macro features
    tickers = ["SPY", "^VIX", "IEF", "HYG"]
    print(f"[SYSTEM] Fetching daily macro data for {tickers} from {start_date} to {end_date}...")
    
    # 2. Download from Yahoo Finance
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    
    # 3. Clean and align data
    data = data.ffill()  # Forward-fill missing values
    data = data.dropna() # Drop any remaining rows with NaN values at the start
    
    # 4. Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.to_csv(save_path)
    
    print(f"[SUCCESS] Macro data saved to {save_path}")
    return data

if __name__ == "__main__":
    print("\n====== M1 RESEARCH NODE: MACRO DATA FETCHER ======")
    
    # Define the 16-year lookback period
    start_date = "2010-01-01"
    end_date = "2026-01-01"
    
    # Execute the fetch
    df = fetch_macro_data(start_date, end_date)

    # Diagnostic Output
    print("\n====== DIAGNOSTIC RESULTS ======")
    print("[SUCCESS] First 5 rows of the dataset:")
    print(df.head())
    print(f"\n[SYSTEM] Final dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("[ACTION] Transfer 'data/raw_macro_data.csv' to the ASUS Execution Node.")