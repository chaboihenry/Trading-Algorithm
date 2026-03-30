# m1_research_node/data_tools/build_backtest_matrix.py

import os
import json
import pandas as pd

def compile_historical_state(lookback_years: int = 3, models_dir: str = "the_models", output_dir: str = "the_execution_node/data"):
    # Compiles the 5-minute historical state strictly for the active trading universe.
    # Extracts directly from the M1 Parquet Vault, bypassing API limits.
    
    print(f"\n====== COMPILING {lookback_years}-YEAR HISTORICAL STATE ======")
    
    # 1. Load the active execution universe
    json_path = os.path.join(models_dir, "curated_universe.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            target_tickers = data.get("flat_list", [])
    except FileNotFoundError:
        print(f"[CRITICAL] {json_path} not found. Ensure the M1 has run its daily discovery.")
        return

    print(f"[SYSTEM] Target Universe: {len(target_tickers)} active assets.")
    
    # 2. Establish time constraints
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=lookback_years)
    cutoff_str = cutoff_dt.strftime('%Y%m%d')
    print(f"[SYSTEM] Extracting data from {cutoff_dt.strftime('%Y-%m-%d')} to Present...")

    ticker_series = {}

    # 3. Extract and align data from the Parquet Vault
    for ticker in target_tickers:
        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
        
        if not os.path.exists(path):
            print(f"  -> [WARNING] Vault path missing for {ticker}. Skipping.")
            continue
            
        print(f"  -> Processing {ticker}...")
        file_dfs = []
        
        # File-by-file extraction to prevent schema crashes
        for file in os.listdir(path):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
                
            if file[:8] < cutoff_str:
                continue
                
            file_path = os.path.join(path, file)
            try:
                df = pd.read_parquet(
                    file_path, 
                    columns=['timestamp', 'price'],
                    filters=[('timestamp', '>=', cutoff_dt)]
                )
                if not df.empty:
                    file_dfs.append(df)
            except Exception:
                continue
                
        if not file_dfs:
            continue
            
        # Concatenate all valid files for this ticker
        df = pd.concat(file_dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').set_index('timestamp')
        
        # Resample to strict 5-minute bars
        resampled_close = df['price'].resample('5min').last()
        ticker_series[ticker] = resampled_close

    if not ticker_series:
        print("[CRITICAL] No valid data extracted from the vault.")
        return

    # 4. Merge all individual series into a single massive DataFrame
    print("\n[SYSTEM] Merging all assets into a unified matrix...")
    master_matrix = pd.DataFrame(ticker_series)
    
    # Forward fill missing ticks (crucial for illiquid assets or halted periods)
    # Then drop any rows at the very beginning that still have NaNs
    master_matrix = master_matrix.ffill().dropna()

    # 5. Export to Parquet
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "backtest_5m_3yr.parquet")
    
    print(f"[SYSTEM] Compressing and writing to {output_filename}...")
    master_matrix.to_parquet(output_filename, engine='pyarrow', compression='snappy')
    
    # 6. Calculate file size for diagnostics
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    
    print("\n====== EXPORT SUCCESS ======")
    print(f"Matrix Shape: {master_matrix.shape[0]} rows (5-min ticks) x {master_matrix.shape[1]} columns")
    print(f"File Size:    {file_size_mb:.2f} MB")
    print(f"[ACTION] Transfer '{output_filename}' to the ASUS Execution Node.")

if __name__ == "__main__":
    compile_historical_state()