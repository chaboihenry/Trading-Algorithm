import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

def _append_discovery_ledger(entry: dict, ledger_path: str = "logs/cluster_discovery_ledger.jsonl"):
    # Append one JSON line per cluster tested. JSONL format (one object per line)
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_universe_list():
    # Decoupled ticker list for scaling
    try:
        with open('universe.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("[CRITICAL] universe.txt not found. Create it in the root directory.")
        return []

def load_daily_from_vault(tickers: list, lookback_days: int = 365):
    # Load daily bars from the Parquet Vault for PCA/DBSCAN clustering.
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
    cutoff_str = cutoff_dt.strftime('%Y%m%d')
    
    daily_prices = {}
    
    for ticker in tickers:
        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"
        if not os.path.exists(path): continue
        
        daily_chunks = []
        for file in sorted(os.listdir(path)):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
            if file[:8] < cutoff_str:
                continue
                
            file_path = os.path.join(path, file)
            try:
                chunk = pd.read_parquet(file_path, columns=['timestamp', 'price'])
                if chunk.empty: continue
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
                daily = chunk.set_index('timestamp')['price'].resample('1D').last()
                daily_chunks.append(daily)
                del chunk
            except Exception:
                continue
        
        if not daily_chunks: continue
        series = pd.concat(daily_chunks).sort_index()
        daily_prices[ticker] = series[~series.index.duplicated(keep='last')].ffill()
    
    if not daily_prices:
        return pd.DataFrame()
    
    return pd.DataFrame(daily_prices).ffill().dropna()

def load_vault_data(cluster_tickers: list, lookback_days: int = 90):
    # Load 5-min bars for cointegration testing with memory-safe chunking
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
    cutoff_str = cutoff_dt.strftime('%Y%m%d')
    
    ticker_series = {}

    for ticker in cluster_tickers:
        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"
        if not os.path.exists(path): continue
        
        resampled_chunks = []
        for file in sorted(os.listdir(path)):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
            if file[:8] < cutoff_str:
                continue
                
            file_path = os.path.join(path, file)
            try:
                chunk = pd.read_parquet(
                    file_path, columns=['timestamp', 'price'],
                    filters=[('timestamp', '>=', cutoff_dt)]
                )
                if chunk.empty: continue
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
                bars = chunk.set_index('timestamp')['price'].resample('5min').last()
                resampled_chunks.append(bars)
                del chunk
            except Exception:
                continue
        
        if not resampled_chunks: continue
        series = pd.concat(resampled_chunks).sort_index()
        ticker_series[ticker] = series[~series.index.duplicated(keep='last')].ffill()
    
    if not ticker_series:
        return pd.DataFrame()
    
    # Align all tickers on shared timestamps
    aligned = pd.DataFrame(ticker_series).dropna()
    return aligned

def test_cointegration(aligned_data: pd.DataFrame, tickers: list):
    # Johansen test to find the mean-reverting spread
    if len(aligned_data) < 156:
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
    # Drop the weakest baskets (longest half-life) until under the limit
    # Shorter half-life = faster mean reversion = stronger signal
    while True:
        unique_tickers = set()
        for data in baskets.values():
            unique_tickers.update(data['tickers'])
            
        if len(unique_tickers) <= max_tickers or not baskets:
            break
        
        # Sort by half-life descending — longest (weakest) gets dropped first
        weakest = max(baskets, key=lambda k: baskets[k].get('half_life', float('inf')))
        print(f"  >> [WEBSOCKET LIMIT] {len(unique_tickers)} tickers. "
              f"Dropping {weakest} (half-life: {baskets[weakest]['half_life']:.2f}d)...")
        baskets.pop(weakest)
        
    return baskets

def run_discovery_pipeline():
    universe = load_universe_list()
    if not universe: return

    print(f"\n=== ASYNC COMPUTE NODE: CLUSTER DISCOVERY ===")
    print(f"--- Processing Research Universe: {len(universe)} Assets ---")
    
    # Load 1 year of daily bars from the vault (replaces yfinance)
    print("[SYSTEM] Loading daily bars from Parquet Vault for PCA clustering...")
    data = load_daily_from_vault(universe, lookback_days=365)
    
    if data.empty or len(data.columns) < 5:
        print("[CRITICAL] Insufficient vault data for clustering.")
        return
    
    print(f"[SYSTEM] Loaded {len(data)} daily bars across {len(data.columns)} assets.")
    
    # Identify economic clusters using PCA and DBSCAN
    returns_t = data.pct_change().dropna().T
    scaled = StandardScaler().fit_transform(returns_t)
    pca = PCA(n_components=min(len(data.columns), 5)).fit_transform(scaled)
    clusters = DBSCAN(eps=1.2, min_samples=2).fit_predict(pca)

    results = pd.DataFrame({'Ticker': returns_t.index, 'Cluster': clusters})
    groups = results[results['Cluster'] != -1].groupby('Cluster')['Ticker'].apply(list)
    
    print(f"[SYSTEM] DBSCAN found {len(groups)} clusters. Testing cointegration...")
   
    confirmed_baskets = {}
    run_timestamp = pd.Timestamp.now(tz='UTC').isoformat()

    for _, cluster_tickers in groups.items():
        aligned = load_vault_data(cluster_tickers)
        if aligned.empty:
            _append_discovery_ledger({
                "timestamp": run_timestamp,
                "tickers": cluster_tickers,
                "status": "no_vault_data",
            })
            continue

        is_coint, hl_days, weights = test_cointegration(aligned, cluster_tickers)

        # Build ledger entry — logs every test, accepted or rejected
        ledger_entry = {
            "timestamp": run_timestamp,
            "tickers": cluster_tickers,
            "is_cointegrated": bool(is_coint),
            "half_life_days": float(hl_days) if hl_days is not None else None,
            "weights": weights if weights else None,
        }

        # Compute hedge ratio sanity metric for the ledger
        if weights:
            abs_weights = [abs(w) for w in weights.values()]
            min_w = min(abs_weights)
            max_w = max(abs_weights)
            ledger_entry["min_max_weight_ratio"] = min_w / max_w if max_w > 0 else 0.0
        else:
            ledger_entry["min_max_weight_ratio"] = None

        # Determine final status
        if not is_coint:
            ledger_entry["status"] = "not_cointegrated"
        elif not (0.01 <= hl_days <= 15.0):
            ledger_entry["status"] = f"half_life_out_of_range ({hl_days:.2f}d)"
        elif ledger_entry["min_max_weight_ratio"] is not None and ledger_entry["min_max_weight_ratio"] < 0.15:
            ledger_entry["status"] = "degenerate_hedge_ratio"
            print(f"  >> [DEGENERATE] {'_'.join(cluster_tickers)}: "
                  f"min/max weight ratio = {ledger_entry['min_max_weight_ratio']:.3f}")
        else:
            spread_name = "_".join(cluster_tickers) + "_Spread"
            confirmed_baskets[spread_name] = {
                'tickers': cluster_tickers,
                'weights': weights,
                'half_life': hl_days
            }
            ledger_entry["status"] = "confirmed"
            print(f"  >> [CONFIRMED] {spread_name} | Half-life: {hl_days:.2f}d | "
                  f"weight ratio: {ledger_entry['min_max_weight_ratio']:.3f}")

        _append_discovery_ledger(ledger_entry)
            
    print(f"\n[SYSTEM] {len(confirmed_baskets)} cointegrated baskets confirmed.")
            
    target_dir = 'the_models'
    os.makedirs(target_dir, exist_ok=True)

    # Save master ledger (preserves historical baskets)
    ledger_path = os.path.join(target_dir, 'universe_baskets.json')
    ledger_payload = {"historical_basket_names": [], "baskets": {}}
    
    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f:
                existing_data = json.load(f)
                if "historical_basket_names" in existing_data and "baskets" in existing_data:
                    ledger_payload = existing_data
                else:
                    ledger_payload["baskets"] = existing_data
        except Exception as e:
            print(f"[WARNING] Could not read existing ledger. Starting fresh. Error: {e}")
            
    current_time = pd.Timestamp.now(tz='UTC').isoformat()
    for basket_name, basket_data in confirmed_baskets.items():
        basket_data['last_seen'] = current_time
        ledger_payload["baskets"][basket_name] = basket_data
        
    ledger_payload["historical_basket_names"] = list(ledger_payload["baskets"].keys())
        
    temp_ledger_path = os.path.join(target_dir, 'universe_baskets_temp.json')
    with open(temp_ledger_path, 'w') as f:
        json.dump(ledger_payload, f, indent=4)
    os.replace(temp_ledger_path, ledger_path)
    
    print(f"[SYSTEM] Master Ledger updated. Tracking {len(ledger_payload['historical_basket_names'])} historical baskets.")

    # Enforce websocket limit — drops weakest (longest half-life) first
    confirmed_baskets = enforce_websocket_limit(confirmed_baskets, max_tickers=30)
    
    approved_tickers = set()
    for data in confirmed_baskets.values():
        approved_tickers.update(data['tickers'])
           
    if approved_tickers:
        payload = {
            "timestamp": current_time,
            "baskets": confirmed_baskets,
            "flat_list": list(approved_tickers)
        }
        
        final_path = os.path.join(target_dir, 'curated_universe.json')
        temp_path = os.path.join(target_dir, 'curated_universe_temp.json')
        
        with open(temp_path, 'w') as f:
            json.dump(payload, f, indent=4)
        os.replace(temp_path, final_path)
            
        print(f"[SUCCESS] Curated {len(approved_tickers)} tickers across {len(confirmed_baskets)} baskets for Live Execution.")
    else:
        print("[WARNING] No baskets survived. Execution node will hold cash.")

if __name__ == "__main__":
    run_discovery_pipeline()