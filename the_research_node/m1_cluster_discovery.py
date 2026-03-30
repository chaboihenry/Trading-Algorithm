import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta, timezone
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

# decouple the universe from the code for scaling
def load_universe_list():
    try:
        with open('universe.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("[CRITICAL] universe.txt not found. Create it in the root directory.")
        return []

def load_vault_data(cluster_tickers: list, lookback_days: int = 90):
    cpu_dataframes = []
    
    # 1. Calculate UTC cutoff for exact filtering
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
    
    # 2. Extract YYYYMMDD for lightning-fast file exclusion
    cutoff_str = cutoff_dt.strftime('%Y%m%d')

    for ticker in cluster_tickers:
        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
        if not os.path.exists(path):
            continue
           
        ticker_dfs = []
        
        # 3. File-by-File iteration bypasses PyArrow's mixed-schema folder crashes entirely
        for file in os.listdir(path):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
                
            # Skip files physically written before our lookback window
            if file[:8] < cutoff_str:
                continue
                
            file_path = os.path.join(path, file)
            try:
                # Pandas safely handles the schema of each file individually in memory
                df = pd.read_parquet(
                    file_path, 
                    columns=['timestamp', 'price'],
                    filters=[('timestamp', '>=', cutoff_dt)]
                )
                if not df.empty:
                    ticker_dfs.append(df)
            except Exception:
                continue
        
        if not ticker_dfs:
            continue
            
        # Concatenate all valid files. Pandas perfectly aligns 'ns' and 'us' without crashing.
        df = pd.concat(ticker_dfs, ignore_index=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').set_index('timestamp')
       
        # Resample to 5-minute bars (78 bars per trading day)
        close_cpu = df['price'].resample('5min').last().ffill()
       
        df_aligned = close_cpu.reset_index()
        df_aligned.columns = ['timestamp', ticker]
        cpu_dataframes.append(df_aligned)

    if not cpu_dataframes: 
        return pd.DataFrame()

    aligned_data = cpu_dataframes[0]
    for df in cpu_dataframes[1:]:
        aligned_data = pd.merge(aligned_data, df, on='timestamp', how='inner')
       
    return aligned_data.dropna().set_index('timestamp')


def test_cointegration(aligned_data: pd.DataFrame, tickers: list):
    # Johansen test to find the mean-reverting 'leash'.
    if len(aligned_data) < 156: # Minimum 2 days of 5-min bars for valid VAR
        return False, None, None

    res = coint_johansen(aligned_data, det_order=0, k_ar_diff=1)
    trace_stat = res.lr1[0]
    crit_95 = res.cvt[0, 1]
   
    if trace_stat > crit_95:
        eigenvector = res.evec[:, 0]
        weights = dict(zip(tickers, eigenvector / eigenvector[0]))
       
        # OU Half-Life Calculation
        spread = aligned_data.dot(eigenvector)
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        ols = sm.OLS(spread_diff, sm.add_constant(spread_lag.loc[spread_diff.index])).fit()
       
        lambda_val = ols.params.iloc[1]
        half_life = -np.log(2) / lambda_val if lambda_val < 0 else np.inf
        return True, (half_life / 78), weights
       
    return False, None, None

def enforce_websocket_limit(baskets: dict, max_tickers: int = 30):
    """
    Recursively drops the lowest-conviction baskets until the total unique
    ticker count across all remaining strategies is <= max_tickers.
    Protects the Alpaca IEX Free Tier websocket connection.
    """
    while True:
        unique_tickers = set()
        for data in baskets.values():
            unique_tickers.update(data['tickers'])
            
        if len(unique_tickers) <= max_tickers:
            break
            
        if not baskets:
            break
            
        # Assuming the baskets dictionary appends newer/weaker clusters last
        # pop the last strategy off the list to free up connection slots
        weakest_strategy = list(baskets.keys())[-1]
        print(f"  >> [WEBSOCKET LIMIT] {len(unique_tickers)} tickers found. Dropping {weakest_strategy}...")
        baskets.pop(weakest_strategy)
        
    return baskets

def run_discovery_pipeline():
    universe = load_universe_list()
    if not universe: return

    print(f"--- Processing Research Universe: {len(universe)} Assets ---")
    data = yf.download(universe, period="1y", interval="1d")['Close'].dropna(axis=1)
    returns_t = data.pct_change().dropna().T
   
    # identify economic clusters using pca and dbscan
    scaled = StandardScaler().fit_transform(returns_t)
    pca = PCA(n_components=min(len(universe), 5)).fit_transform(scaled)
    clusters = DBSCAN(eps=1.2, min_samples=2).fit_predict(pca)

    results = pd.DataFrame({'Ticker': returns_t.index, 'Cluster': clusters})
    groups = results[results['Cluster'] != -1].groupby('Cluster')['Ticker'].apply(list)
   
    confirmed_baskets = {}

    for _, cluster_tickers in groups.items():
        aligned = load_vault_data(cluster_tickers)
        if aligned.empty: continue
           
        is_coint, hl_days, weights = test_cointegration(aligned, cluster_tickers)

        if is_coint and (0.01 <= hl_days <= 15.0):
            # create a unique identifier for the spread
            spread_name = "_".join(cluster_tickers) + "_Spread"
            
            confirmed_baskets[spread_name] = {
                'tickers': cluster_tickers, 
                'weights': weights,
                'half_life': hl_days
            }
            
    # ENFORCE ALPACA WEBSOCKET LIMIT BEFORE SAVING
    confirmed_baskets = enforce_websocket_limit(confirmed_baskets, max_tickers=30)
    
    # Re-calculate approved tickers after enforcing the limit
    approved_tickers = set()
    for data in confirmed_baskets.values():
        approved_tickers.update(data['tickers'])
           
    if approved_tickers:
        payload = {
            "timestamp": pd.Timestamp.now(tz='UTC').isoformat(),
            "baskets": confirmed_baskets,
            "flat_list": list(approved_tickers)
        }
        
        target_dir = 'the_models'
        os.makedirs(target_dir, exist_ok=True)
        
        # Define paths for the atomic swap
        final_path = os.path.join(target_dir, 'curated_universe.json')
        temp_path = os.path.join(target_dir, 'curated_universe_temp.json')
        
        # Write the heavy payload to the temporary file
        # keeps the final file untouched during the I/O process
        with open(temp_path, 'w') as f:
            json.dump(payload, f, indent=4)
            
        # Instantly swap the temp file to the final destination
        # This is an atomic POSIX operation, meaning zero read downtime
        os.replace(temp_path, final_path)
            
        print(f"[SUCCESS] Curated {len(approved_tickers)} tickers for Live Execution.")

if __name__ == "__main__":
    run_discovery_pipeline()