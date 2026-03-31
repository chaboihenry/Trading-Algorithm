import os
import pandas as pd

def audit_parquet_columns(models_dir: str = "the_models"):
    print("\n====== DIAGNOSTIC: VAULT SCHEMA AUDIT ======")
    
    # 1. Select a few highly liquid sample tickers to check
    sample_tickers = ['AAPL', 'MSFT', 'SPY']
    print(f"[SYSTEM] Auditing schemas for samples: {sample_tickers}")
    
    for ticker in sample_tickers:
        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
        
        if not os.path.exists(path):
            print(f"  -> [WARNING] Vault path missing for {ticker}. Skipping.")
            continue
            
        # 2. Find the first valid Parquet file in the directory
        target_file = None
        for file in os.listdir(path):
            if file.endswith('.parquet') and not file.startswith('._'):
                target_file = os.path.join(path, file)
                break
                
        if not target_file:
            print(f"  -> [WARNING] No valid Parquet files found inside {ticker} folder.")
            continue
            
        # 3. Read ONLY the columns (to save RAM and time) and print them
        try:
            # By not specifying 'columns', we force Pandas to load all available headers
            df = pd.read_parquet(target_file)
            columns = df.columns.tolist()
            
            print(f"\n[SUCCESS] {ticker} Schema Confirmed:")
            print(f"  -> File Inspected: {os.path.basename(target_file)}")
            print(f"  -> Columns Found:  {columns}")
            
            # 4. Specifically check for our missing microstructure features
            missing = []
            for req in ['volume', 'bid_price', 'ask_price']:
                if req not in columns:
                    missing.append(req)
                    
            if missing:
                print(f"  *** [ALERT] Missing required columns: {missing}")
            else:
                print(f"  *** [VERIFIED] All microstructure columns are present!")
                
        except Exception as e:
            print(f"  -> [ERROR] Failed to read {ticker} file: {e}")

if __name__ == "__main__":
    audit_parquet_columns()