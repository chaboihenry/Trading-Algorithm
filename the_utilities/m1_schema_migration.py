import os
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Define the unbreakable nanosecond schema
STRICT_SCHEMA = pa.schema([
    ('timestamp', pa.timestamp('ns', tz='UTC')),
    ('price', pa.float64()),
    ('size', pa.float64()),
    ('exchange', pa.string())
])

def migrate_vault_schema():
    # Target the primary storage directory
    base_dir = "/Volumes/Vault/quant_data/tick data storage"
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Directory not found: {base_dir}")
        return

    tickers = os.listdir(base_dir)
    print(f"Starting schema migration for {len(tickers)} assets...")

    for ticker in tickers:
        parquet_dir = os.path.join(base_dir, ticker, "parquet")
        
        if not os.path.isdir(parquet_dir):
            continue
            
        files = os.listdir(parquet_dir)
        
        for file in files:
            if not file.endswith(".parquet") or file.startswith("._"):
                continue
                
            file_path = os.path.join(parquet_dir, file)
            temp_file_path = file_path + ".tmp"
            
            try:
                df = pd.read_parquet(file_path)
            except Exception as e:
                print(f"[WARNING] Could not read {file_path}: {e}")
                continue
                
            if df.empty:
                continue
            
            # --- COLUMN NORMALIZATION ---
            # Rename backfilled columns to match the live scraper
            if 'close' in df.columns and 'price' not in df.columns:
                df.rename(columns={'close': 'price'}, inplace=True)
            if 'volume' in df.columns and 'size' not in df.columns:
                df.rename(columns={'volume': 'size'}, inplace=True)
                
            # Inject missing columns to satisfy the strict schema
            if 'price' not in df.columns:
                df['price'] = np.nan
            if 'size' not in df.columns:
                df['size'] = 0.0
            if 'exchange' not in df.columns:
                df['exchange'] = "HISTORICAL"
                
            # Ensure proper datatypes before PyArrow casting
            df['price'] = df['price'].astype(float)
            df['size'] = df['size'].astype(float)
            df['exchange'] = df['exchange'].astype(str)
            
            # Standardize the timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            
            # --- SCHEMA ENFORCEMENT ---
            try:
                table = pa.Table.from_pandas(df, schema=STRICT_SCHEMA)
                pq.write_table(table, temp_file_path, compression='zstd')
                os.replace(temp_file_path, file_path)
            except Exception as e:
                print(f"[ERROR] Failed to migrate {file_path}: {e}")
                continue
            
        print(f"[SUCCESS] Migrated schema for: {ticker}")

    print("=== Vault Schema Migration Complete ===")

if __name__ == "__main__":
    migrate_vault_schema()