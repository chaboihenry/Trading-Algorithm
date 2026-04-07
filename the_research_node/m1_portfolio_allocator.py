import os
import json
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

UNIVERSE_PATH = "the_models/curated_universe.json"

# --- LÓPEZ DE PRADO HRP MATHEMATICS ---

def get_ivp(cov, **kargs):
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def get_cluster_var(cov, c_items):
    cov_ = cov.loc[c_items, c_items] 
    w_ = get_ivp(cov_).reshape(-1, 1)
    c_var = np.dot(np.dot(w_.T, cov_), w_)[0, 0]
    return c_var

def get_quasi_diag(link):
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3] 
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2) 
        df0 = sort_ix[sort_ix >= num_items] 
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0] 
        df0 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df0]) 
        sort_ix = sort_ix.sort_index() 
        sort_ix.index = range(sort_ix.shape[0]) 
    return sort_ix.tolist()

def get_rec_bipart(cov, sort_ix):
    w = pd.Series(1, index=sort_ix, dtype=float)
    c_items = [sort_ix] 
    while len(c_items) > 0:
        c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1] 
        for i in range(0, len(c_items), 2): 
            c_items0 = c_items[i] 
            c_items1 = c_items[i + 1] 
            c_var0 = get_cluster_var(cov, c_items0)
            c_var1 = get_cluster_var(cov, c_items1)
            alpha = 1 - c_var0 / (c_var0 + c_var1)
            w[c_items0] *= alpha 
            w[c_items1] *= 1 - alpha 
    return w

def correl_dist(corr):
    dist = ((1 - corr) / 2.)**.5 
    return dist

# --- MAIN ALLOCATION PIPELINE ---

def run_hrp_allocation():
    print("\n=== ASYNC COMPUTE NODE: HRP PORTFOLIO ALLOCATOR ===")
    
    if not os.path.exists(UNIVERSE_PATH):
        print("[ERROR] Curated universe missing. Cannot allocate capital.")
        return

    with open(UNIVERSE_PATH, 'r') as f:
        universe_data = json.load(f)
    
    baskets = universe_data.get("baskets", {})
    if len(baskets) < 2:
        print("[WARNING] Not enough baskets for HRP. Defaulting to equal weight.")
        for name in baskets:
            baskets[name]['capital_allocation'] = 1.0 / len(baskets) if baskets else 1.0
        universe_data['baskets'] = baskets
        with open(UNIVERSE_PATH, 'w') as f:
            json.dump(universe_data, f, indent=4)
        return

    spread_returns = {}

    # 1. Reconstruct historical returns for each active spread
    for name, data in baskets.items():
        tickers = data['tickers']
        weights = data['weights']
        
        prices = {}
        for t in tickers:
            path = f"/Volumes/Vault/quant_data/tick data storage/{t}/parquet/training_data"
            if not os.path.exists(path): continue
            
            f# Filter out the massive merged files and WRDS backups to prevent RAM exhaustion
            files = [os.path.join(path, f) for f in os.listdir(path) 
                     if f.endswith('.parquet') 
                     and not f.startswith('._')
                     and "merged" not in f
                     and f != "ticks.parquet"
                     and "wrds" not in f]
            
            if not files: continue
            
            # Load all data to bypass alphabetical sorting mismatches
            dfs = []
            for f in files:
                try:
                    dfs.append(pd.read_parquet(f, columns=['timestamp', 'price']))
                except Exception:
                    continue
                    
            if not dfs: continue
            df_t = pd.concat(dfs, ignore_index=True)
            df_t['timestamp'] = pd.to_datetime(df_t['timestamp'], utc=True)
            
            # Resample to daily returns for portfolio covariance stability
            prices[t] = df_t.set_index('timestamp')['price'].resample('1D').last().ffill()
            
        if len(prices) != len(tickers): 
            print(f"  >> [WARNING] Missing data for {name}. Skipping in allocation.")
            continue
            
        # Safely align this basket's prices
        basket_df = pd.DataFrame(prices).ffill().dropna()
        if basket_df.empty: continue
            
        # Calculate the daily value of the spread
        spread_val = sum(basket_df[t] * weights[t] for t in tickers)
        spread_returns[name] = spread_val.pct_change().dropna()

    if len(spread_returns) < 2:
        print("[ERROR] Insufficient spread return data for HRP calculation.")
        return

    # 2. Build Covariance Matrix of the Spreads safely
    returns_df = pd.DataFrame(spread_returns)
    
    # Forward fill missing days across spreads, then drop the initial NaN rows
    returns_df = returns_df.ffill().dropna()
    
    # Protect against divide-by-zero: drop strategies with strictly zero variance
    returns_df = returns_df.loc[:, returns_df.std() > 0]
    
    if len(returns_df) < 30 or len(returns_df.columns) < 2:
        print("[WARNING] Insufficient overlapping variance for HRP. Equal weighting fallback.")
        for name in baskets:
            baskets[name]['capital_allocation'] = 1.0 / len(baskets)
    else:
        cov, corr = returns_df.cov(), returns_df.corr()
        print(f"  >> Calculating HRP allocation for {len(returns_df.columns)} active strategies...")

        # 3. Apply Hierarchical Risk Parity
        dist = correl_dist(corr)
        dist_array = squareform(dist.values, checks=False) 
        link = sch.linkage(dist_array, 'single')
        
        sort_ix = get_quasi_diag(link)
        sort_ix = corr.index[sort_ix].tolist()
        
        cov = cov.loc[sort_ix, sort_ix]
        hrp_weights = get_rec_bipart(cov, sort_ix)
        
        # 4. Inject allocations back into the JSON payload
        for name in baskets:
            if name in hrp_weights:
                baskets[name]['capital_allocation'] = round(float(hrp_weights[name]), 4)
                print(f"  >> {name}: {baskets[name]['capital_allocation'] * 100:.2f}%")
            else:
                baskets[name]['capital_allocation'] = 0.0

    universe_data['baskets'] = baskets
    
    # Atomic Write to protect the live scraper
    temp_path = UNIVERSE_PATH + ".tmp"
    with open(temp_path, 'w') as f:
        json.dump(universe_data, f, indent=4)
    os.replace(temp_path, UNIVERSE_PATH)

    print("\n[SUCCESS] Capital Allocation complete and injected into curated_universe.json.")

if __name__ == "__main__":
    run_hrp_allocation()