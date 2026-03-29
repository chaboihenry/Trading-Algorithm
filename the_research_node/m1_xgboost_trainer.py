import os
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from numba import jit
import pyarrow.dataset as ds
import pyarrow as pa
import pyarrow.compute as pc
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import json
import glob


# fetch level 1 order book data for microstructure features
def get_alpaca_microstructure(symbol: str, start_dt, end_dt):
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)
    
    request_params = StockQuotesRequest(symbol_or_symbols=symbol, start=start_dt, end=end_dt)
    
    print(f"Fetching Level 1 Quotes for {symbol}...")
    quotes = client.get_stock_quotes(request_params).df
    
    # Alpaca returns a MultiIndex (symbol, timestamp). Drop 'symbol' so resample works.
    quotes = quotes.reset_index(level='symbol', drop=True)
    
    # filter out bad data from the exchange
    quotes = quotes[(quotes['ask_price'] > 0) & (quotes['bid_price'] > 0)]
    quotes = quotes[(quotes['ask_size'] > 0) & (quotes['bid_size'] > 0)]
    
    # calculate the true microstructure features using order book depth
    quotes['mid_price'] = (quotes['ask_price'] + quotes['bid_price']) / 2.0
    quotes['micro_price'] = (
        (quotes['bid_price'] * quotes['ask_size']) + (quotes['ask_price'] * quotes['bid_size'])
    ) / (quotes['bid_size'] + quotes['ask_size'])
    
    quotes['micro_price_divergence'] = quotes['micro_price'] - quotes['mid_price']
    quotes['relative_spread'] = (quotes['ask_price'] - quotes['bid_price']) / quotes['mid_price']
    
    # resample to 5-minute bars to match the vault tick data
    micro_features = quotes[['micro_price_divergence', 'relative_spread']].resample('5min').mean().ffill()
    
    return micro_features

# compile the path-dependent loop into raw ARM machine code for the M1 CPU
@jit(nopython=True)
def sample_imbalance_bars(signed_dv_array, threshold):
    # pre-allocate the maximum possible array size for speed
    bar_indices = np.empty(len(signed_dv_array), dtype=np.int64)
    count = 0
    theta = 0.0

    for i in range(len(signed_dv_array)):
        theta += signed_dv_array[i]

        # imbalance breaches threshold, sample the bar and RESET
        if abs(theta) >= threshold:
            bar_indices[count] = i
            count += 1
            theta = 0.0 # path-dependent reset
    
    return bar_indices[:count]

# construct the bars using Apple Silicon Unified Memory
def construct_m1_dibs(ticker: str, threshold: float = 50_000_000):
    print(f"Loading Vault ticks for {ticker} into Unified Memory...")
    
    path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
    if not os.path.exists(path):
        return pd.DataFrame()
        
    try:
        # filter out the Apple '._' ghost files natively
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet') and not f.startswith('._')]
        dfs = []
        
        for f in files:
            try:
                df_chunk = pd.read_parquet(f, columns=['timestamp', 'price', 'size'])
                
                # Strip timezones and force nanosecond resolution on the isolated chunk
                if df_chunk['timestamp'].dt.tz is not None:
                    df_chunk['timestamp'] = df_chunk['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                
                df_chunk['timestamp'] = df_chunk['timestamp'].astype('datetime64[ns]')
                dfs.append(df_chunk)
            except Exception as e:
                print(f"  >> [WARNING] Skipping corrupted chunk {os.path.basename(f)}: {e}")
                
        if not dfs:
            print(f"[ERROR] No valid data chunks found for {ticker}.")
            return pd.DataFrame()
            
        # Stitch the standardized chunks together in CPU RAM
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values('timestamp').reset_index(drop=True)

        print(f"Calculating tick direction and signed volume for {ticker}...")
        df['dollar_volume'] = df['price'] * df['size']
        df['price_change'] = df['price'].diff()
        
        # tick rule: +1 for uptick, -1 for downtick, forward fill zeros
        df['tick_direction'] = np.sign(df['price_change']).replace(0, np.nan).ffill().fillna(1)
        df['signed_dollar_volume'] = df['dollar_volume'] * df['tick_direction']

        print(f"Executing C-compiled sampling loop (Threshold: ${threshold:,.0f})...")
        # extract raw numpy array for Numba
        signed_dv_np = df['signed_dollar_volume'].values.astype(np.float64)
        sampled_indices = sample_imbalance_bars(signed_dv_np, threshold)
        print(f"   -> Extracted {len(sampled_indices)} Dollar Imbalance Bars.")

        # group the ticks into OHLCV bars based on the sampled indices
        group_flags = np.zeros(len(df), dtype=np.int32)
        group_flags[sampled_indices] = 1
        group_ids = np.cumsum(group_flags)
        
        # shift group IDs so the triggering tick is included in the current bar
        group_ids = np.roll(group_ids, 1)
        group_ids[0] = 0
        df['group_id'] = group_ids

        print("Aggregating Microstructure features...")
        # standard Pandas groupby is highly optimized for ARM CPUs
        dibs = df.groupby('group_id').agg(
            timestamp=('timestamp', 'last'),
            open=('price', 'first'),
            high=('price', 'max'),
            low=('price', 'min'),
            close=('price', 'last'),
            volume=('size', 'sum'),
            dollar_volume=('dollar_volume', 'sum')
        )
        
        dibs = dibs.set_index('timestamp')
        return dibs

    except Exception as e:
        print(f"[ERROR] DIB Construction Failed: {e}")
        return pd.DataFrame()
    
# calculates weights for fixed-width window fractional differentiation
def get_weights_ffd(d: float, threshold: float = 1e-5):
    weights = [1.0]
    k = 1
    while True:
        next_weight = -weights[-1] * (d - k + 1) / k
        if abs(next_weight) < threshold:
            break
        weights.append(next_weight)
        k += 1
    return np.array(weights).reshape(-1, 1)

# applies fractional differentiation to retain memory while achieving stationarity
def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5): 
    weights = get_weights_ffd(d, threshold)
    weights = weights[::-1] # reverse so most recent price gets weight of 1.0

    def dot_product(x):
        return np.dot(x, weights)[0]
    
    df = series.to_frame('close')
    # apply rolling dot product across the price series
    df['frac_diff'] = df['close'].rolling(window=len(weights)).apply(dot_product, raw=True)

    return df['frac_diff'].dropna()

# calculates dynamic volatility to size profit/stop-loss barriers
def get_daily_volatility(close_series: pd.Series, span0: int = 100):
    returns = close_series.pct_change()
    ewma_vol = returns.ewm(span=span0).std()
    # shift forward by 1 to prevent look-ahead bias
    return ewma_vol.shift(1).dropna()

# evaluates historical trades hitting profit-take, stop-loss, or time-out
def apply_triple_barrier(spread_prices: pd.Series, events: pd.DataFrame, pt_sl: list, max_bars: int = 20):
    out = pd.DataFrame(index=events.index, columns=['pt', 'sl', 't1', 'ret', 'bin'])
    
    for loc, target_vol in events['trgt'].items():
        # Look forward up to 'max_bars' into the future
        start_idx = spread_prices.index.get_loc(loc)
        end_idx = min(start_idx + max_bars, len(spread_prices))
        path = spread_prices.iloc[start_idx:end_idx]
        
        if len(path) < 2:
            continue
            
        path_ret = (path / path.iloc[0]) - 1
        
        # Calculate dynamic barrier levels based on local volatility
        pt_bound = pt_sl[0] * target_vol
        sl_bound = -pt_sl[1] * target_vol
        
        pt_touches = path_ret[path_ret > pt_bound].index
        sl_touches = path_ret[path_ret < sl_bound].index
        
        pt_time = pt_touches[0] if len(pt_touches) > 0 else pd.NaT
        sl_time = sl_touches[0] if len(sl_touches) > 0 else pd.NaT
        t1_time = path.index[-1] # Vertical barrier (time out)
        
        # Determine which barrier was hit first
        first_touch = pd.Series({'pt': pt_time, 'sl': sl_time, 't1': t1_time}).dropna().min()
        
        out.loc[loc, 'ret'] = path_ret.loc[first_touch]
        
        # Label 1 for profit, 0 for loss or time-out
        if first_touch == pt_time:
            out.loc[loc, 'bin'] = 1
        else:
            out.loc[loc, 'bin'] = 0
            
    return out.dropna(subset=['bin'])

# prevents data leakage by removing overlapping training periods
def custom_purged_kfold(times: pd.Series, n_splits: int = 3, embargo_pct: float = 0.01):
    indices = np.arange(len(times))
    test_splits = np.array_split(indices, n_splits)
    embargo_length = int(len(times) * embargo_pct)
    
    splits = []
    for test_idx in test_splits:
        test_start = test_idx[0]
        test_end = test_idx[-1] + embargo_length
        
        # Purge training data that touches the test or embargo window
        train_idx = np.concatenate([
            indices[:max(0, test_start)],
            indices[min(len(indices), test_end):]
        ])
        splits.append((train_idx, test_idx))
    return splits

def train_meta_labeler():
    print("\n=== ASYNC COMPUTE NODE: XGBOOST META-LABELER ===")
    MODELS_DIR = "the_models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # In production, this reads the exact weights from the discovery engine's output
    # Mocking the V/MA spread for this run
    mock_pairs = [{'t1': 'V', 't2': 'MA', 'w1': 1.0, 'w2': -0.5412, 'hl': 12}]
    
    all_features = []
    all_labels = []
    
    for pair in mock_pairs:
        print(f"\nProcessing {pair['t1']}/{pair['t2']} Spread...")
        
        # 1. Load DIBs (Using Ticker 1 as the primary microstructure driver)
        dibs = construct_m1_dibs(pair['t1'], threshold=10_000_000)
        if dibs.empty: continue
            
        # 2. Reconstruct the synthetic spread price using the DIB timestamps
        # (Assuming 'close' represents the anchor price)
        spread_price = dibs['close'] * pair['w1'] # Simplified spread anchor
        
        # 3. Generate Primary Signals (Z-Scores)
        window = int(pair['hl'] * 2)
        mean = spread_price.rolling(window).mean()
        std = spread_price.rolling(window).std()
        z_score = (spread_price - mean) / std
        
        # 4. Identify entry events (Z-score > 2 or < -2)
        entry_events = pd.DataFrame({'z_score': z_score})
        entry_events = entry_events[(entry_events['z_score'] > 2.0) | (entry_events['z_score'] < -2.0)].copy()
        
        # 5. Calculate Dynamic Volatility
        volatility = get_daily_volatility(spread_price)
        entry_events['trgt'] = volatility.reindex(entry_events.index)
        entry_events = entry_events.dropna()
        
        if entry_events.empty: continue
        print(f"  >> Extracted {len(entry_events)} primary historical signals.")
        
        # 6. Apply Triple Barrier
        labels = apply_triple_barrier(spread_price, entry_events, pt_sl=[2.0, 1.0])
        print(f"  >> Win Rate: {(labels['bin'].mean() * 100):.1f}%")
        
        # 7. Apply Fractional Differentiation
        frac_diff = frac_diff_ffd(spread_price, d=0.4)
        
        # 8. Fetch Alpaca Microstructure (Fetching 3 years of quotes takes time)
        # fetch the last 30 days of the events to respect Alpaca API limits during training
        start_quote = entry_events.index[-1] - pd.Timedelta(days=30)
        end_quote = entry_events.index[-1]
        microstructure = get_alpaca_microstructure(pair['t1'], start_quote, end_quote)
        
        # 9. Build the Final Feature Space for the execution computer (ASUS)
        features = pd.DataFrame(index=labels.index)
        features['frac_diff'] = frac_diff.reindex(labels.index)
        features['volatility'] = entry_events['trgt']
        features['Z_Score'] = entry_events['z_score']
        
        # Target Position: 1 if Z < -2 (Long Spread), -1 if Z > 2 (Short Spread)
        features['Target_Position'] = np.where(features['Z_Score'] < -2.0, 1, -1)
        
        features['micro_price_divergence'] = microstructure['micro_price_divergence'].reindex(labels.index).ffill().fillna(0)
        features['relative_spread'] = microstructure['relative_spread'].reindex(labels.index).ffill().fillna(0)
        
        all_features.append(features)
        all_labels.append(labels['bin'].astype(int))
        
    if not all_features:
        print("[WARNING] Insufficient data to train Meta-Labeler.")
        return
        
    X = pd.concat(all_features).dropna()
    y = pd.concat(all_labels).reindex(X.index)
    
    print(f"\nTraining XGBoost on {len(X)} historical setups...")
    
    # 10. Train with Apple Silicon optimizations
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        n_jobs=-1,  
        random_state=42
    )
    
    cv_splits = custom_purged_kfold(pd.Series(X.index), n_splits=3)
    
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100]
    }
    
    search = RandomizedSearchCV(base_model, param_grid, cv=cv_splits, scoring='roc_auc', n_iter=5, verbose=0)
    search.fit(X, y)
    
    best_model = search.best_estimator_
    print(f"\n[SUCCESS] Model Optimized. ROC-AUC: {search.best_score_:.4f}")
    
    # 11. Versioning System
    existing_models = glob.glob(os.path.join(MODELS_DIR, "meta_labeler_v*.json"))
    version = 1
    if existing_models:
        versions = [int(m.split("_v")[-1].split(".")[0]) for m in existing_models if "_v" in m]
        version = max(versions) + 1 if versions else 1
        
    model_name = f"meta_labeler_v{version}.json"
    model_path = os.path.join(MODELS_DIR, model_name)
    
    best_model.save_model(model_path)
    print(f"[SAVED] {model_name} securely exported to {MODELS_DIR}/")
    
    with open(os.path.join(MODELS_DIR, "active_model_version.txt"), "w") as f:
        f.write(model_name)


if __name__ == "__main__":
    train_meta_labeler()






