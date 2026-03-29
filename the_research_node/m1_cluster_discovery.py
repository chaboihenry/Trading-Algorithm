import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
import pyarrow.dataset as ds
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

def load_vault_data(cluster_tickers: list):
    # reads ZSTD compressed parquet directories from the Vault SSD.
    cpu_dataframes = []
   
    for ticker in cluster_tickers:

        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
       
        if not os.path.exists(path):
            continue
           
        try:
            # ds.dataset seamlessly handles thousands of small compressed files
            dataset = ds.dataset(path, format="parquet")
            df = dataset.to_table(columns=['timestamp', 'price']).to_pandas()
           
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
           
            # Resample to 5-minute bars (78 bars per trading day)
            close_cpu = df['price'].resample('5min').last().ffill()
           
            df_aligned = close_cpu.reset_index()
            df_aligned.columns = ['timestamp', ticker]
            cpu_dataframes.append(df_aligned)
        except Exception as e:
            print(f"  >> [ERROR] {ticker}: {e}")
            return pd.DataFrame()

    if not cpu_dataframes: return pd.DataFrame()

    # inner join to ensure only trade when all assets in the basket have data
    aligned_data = cpu_dataframes[0]
    for df in cpu_dataframes[1:]:
        aligned_data = pd.merge(aligned_data, df, on='timestamp', how='inner')
       
    return aligned_data.dropna().set_index('timestamp')

def test_cointegration(aligned_data: pd.DataFrame, tickers: list):
    """Johansen test to find the mean-reverting 'leash'."""
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

def run_discovery_pipeline():
    universe = load_universe_list()
    if not universe: return

    print(f"--- Processing Research Universe: {len(universe)} Assets ---")
    data = yf.download(universe, period="1y", interval="1d")['Close'].dropna(axis=1)
    returns_t = data.pct_change().dropna().T
   
    # Identify economic clusters using PCA + DBSCAN
    scaled = StandardScaler().fit_transform(returns_t)
    pca = PCA(n_components=min(len(universe), 5)).fit_transform(scaled)
    clusters = DBSCAN(eps=1.2, min_samples=2).fit_predict(pca)

    results = pd.DataFrame({'Ticker': returns_t.index, 'Cluster': clusters})
    groups = results[results['Cluster'] != -1].groupby('Cluster')['Ticker'].apply(list)
   
    confirmed_baskets = []
    approved_tickers = set()

    for _, cluster_tickers in groups.items():
        aligned = load_vault_data(cluster_tickers)
        if aligned.empty: continue
           
        is_coint, hl_days, weights = test_cointegration(aligned, cluster_tickers)

        if is_coint and (0.01 <= hl_days <= 15.0):
            confirmed_baskets.append({'basket': cluster_tickers, 'weights': weights})
            approved_tickers.update(cluster_tickers)
           
    if approved_tickers:
        with open('curated_universe.json', 'w') as f:
            json.dump(list(approved_tickers), f, indent=4)
        print(f"[SUCCESS] Curated {len(approved_tickers)} tickers for Live Execution.")

if __name__ == "__main__":
    run_discovery_pipeline()