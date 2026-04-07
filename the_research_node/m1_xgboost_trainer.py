import os
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from numba import jit
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest

# --- CONFIGURATION & ENV ---
load_dotenv()
API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
UNIVERSE_PATH = "the_models/curated_universe.json"
MODELS_DIR = "the_models"

# --- 1. DIB CONSTRUCTION ENGINE (APPLE SILICON OPTIMIZED) ---
@jit(nopython=True)
def sample_imbalance_bars(signed_dv_array, threshold):
    # Compiles path-dependent loop into raw ARM machine code
    bar_indices = np.empty(len(signed_dv_array), dtype=np.int64)
    count = 0
    theta = 0.0

    for i in range(len(signed_dv_array)):
        theta += signed_dv_array[i]
        if abs(theta) >= threshold:
            bar_indices[count] = i
            count += 1
            theta = 0.0 # Path-dependent reset
    
    return bar_indices[:count]

def construct_m1_dibs(ticker: str, threshold: float = 50_000_000):
    print(f"  >> Constructing Structural DIBs for Anchor Ticker: {ticker}...")
    path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet"
    
    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if f.endswith(".parquet") and not f.startswith("._")
    ]
    if not files: 
        return pd.DataFrame()
        
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f, columns=['timestamp', 'price', 'size']))
        except Exception:
            continue
            
    if not dfs: return pd.DataFrame()
    
    df = pd.concat(dfs, ignore_index=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    df['dollar_volume'] = df['price'] * df['size']
    df['tick_direction'] = np.sign(df['price'].diff()).replace(0, np.nan).ffill().fillna(1)
    df['signed_dollar_volume'] = df['dollar_volume'] * df['tick_direction']

    signed_dv_np = df['signed_dollar_volume'].values.astype(np.float64)
    sampled_indices = sample_imbalance_bars(signed_dv_np, threshold)
    
    group_flags = np.zeros(len(df), dtype=np.int32)
    group_flags[sampled_indices] = 1
    group_ids = np.cumsum(group_flags)
    group_ids = np.roll(group_ids, 1)
    group_ids[0] = 0
    df['group_id'] = group_ids

    dibs = df.groupby('group_id').agg(
        timestamp=('timestamp', 'last'),
        close=('price', 'last')
    ).set_index('timestamp')
    
    return dibs

# --- 2. MICROSTRUCTURE ENGINE (OFFLINE LOCAL COMPUTE) ---
def get_offline_microstructure(df_trades):
    # Calculates order flow and liquidity proxies directly from your SSD Vault trades
    # bypassing the extremely slow Alpaca Quotes API.
    try:
        df = df_trades.copy()
        
        # 1. Order Flow Imbalance (OFI proxy via trades)
        df['price_change'] = df['price'].diff()
        df['tick_dir'] = np.sign(df['price_change']).replace(0, np.nan).ffill().fillna(1)
        df['signed_vol'] = df['size'] * df['tick_dir']
        
        # 2. Roll's Measure (Effective Spread Proxy)
        # Roll's model states that the spread is related to the serial covariance of price changes
        covar = df['price_change'].rolling(30).cov(df['price_change'].shift(1))
        df['roll_measure'] = 2 * np.sqrt(np.abs(covar))
        
        # Resample to match the structural timeframe
        micro = pd.DataFrame()
        micro['order_flow_imbalance'] = df['signed_vol'].rolling(50).sum()
        micro['effective_spread'] = df['roll_measure']
        
        return micro.dropna()
    except Exception as e:
        print(f"  >> [WARNING] Offline Microstructure failed: {e}")
        return pd.DataFrame()

# --- 3. FRACTIONAL DIFFERENTIATION (Memory Preservation Engine) ---
def get_weights_ffd(d: float, threshold: float = 1e-4):
    w = [1.0]
    k = 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < threshold:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def apply_frac_diff(series, d, threshold=1e-4):
    weights = get_weights_ffd(d, threshold)
    if len(weights) > len(series):
        return pd.Series(dtype=float) 
    res = series.rolling(len(weights)).apply(lambda x: np.dot(x, weights.flatten()), raw=True)
    return res.dropna()

def find_optimal_d(series):
    # Searches for minimum 'd' value that achieves ADF stationarity (< 0.05)
    for d in np.linspace(0, 1, 11):
        fd = apply_frac_diff(series, d)
        if fd.empty or len(fd) < 30: 
            continue
        try:
            if adfuller(fd)[1] < 0.05:
                return d, fd
        except Exception:
            continue
    return 1.0, series.diff().dropna()

# --- 4. TRIPLE BARRIER LABELING ENGINE ---
def get_daily_vol(close, span0=100):
    # Enforce strictly unique indices to prevent alignment crashes
    close = close[~close.index.duplicated(keep='last')]
    
    # Find index locations of timestamps 1 day prior
    idx_1day_ago = close.index.searchsorted(close.index - pd.Timedelta(days=1))
    valid = idx_1day_ago > 0
    
    # Map the current and past timestamps
    current_ts = close.index[valid]
    past_ts = close.index[idx_1day_ago[valid] - 1]
    
    # Calculate returns using isolated numpy arrays to bypass Pandas alignment issues
    ret = pd.Series(
        close.loc[current_ts].values / close.loc[past_ts].values - 1,
        index=current_ts
    )
    return ret.ewm(span=span0).std().dropna()

def apply_triple_barrier(prices, events, pt_sl=[2, 1], t1=5):
    # pt_sl: [Profit Take Multiplier, Stop Loss Multiplier]
    # t1: Vertical barrier in hours (Time-Out)
    out = events[['trgt']].copy()
    
    out['bin'] = 0
    for loc, trgt in events['trgt'].items():
        subset = prices[loc:loc + pd.Timedelta(hours=t1)]
        if subset.empty: continue
        
        ret = (subset / prices[loc]) - 1
        pt = trgt * pt_sl[0]
        sl = -trgt * pt_sl[1]
        
        if (ret > pt).any(): out.at[loc, 'bin'] = 1  # Hit Profit
        elif (ret < sl).any(): out.at[loc, 'bin'] = 0 # Hit Stop Loss
        else: out.at[loc, 'bin'] = 0                # Time-Out 
    return out

# --- 5. PURGED K-FOLD CV (Prevents Data Leakage) ---
def custom_purged_kfold(times: pd.Series, n_splits: int = 3, embargo_pct: float = 0.01):
    indices = np.arange(len(times))
    test_splits = np.array_split(indices, n_splits)
    embargo_length = int(len(times) * embargo_pct)
    
    splits = []
    for test_idx in test_splits:
        test_start = test_idx[0]
        test_end = test_idx[-1] + embargo_length
        train_idx = np.concatenate([
            indices[:max(0, test_start)],
            indices[min(len(indices), test_end):]
        ])
        splits.append((train_idx, test_idx))
    return splits

# --- 6. MAIN META-LABELER PIPELINE ---
def train_meta_labeler():
    print("\n=== ASYNC COMPUTE NODE: XGBOOST META-LABELER ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if not os.path.exists(UNIVERSE_PATH):
        print("[ERROR] Curated universe missing. Run discovery pipeline first.")
        return

    with open(UNIVERSE_PATH, 'r') as f:
        universe_data = json.load(f)
    
    baskets = universe_data.get("baskets", {})
    all_X, all_y = [], []

    for name, data in baskets.items():
        print(f"\nProcessing Spread: {name}")
        tickers = data['tickers']
        weights = data['weights']
        anchor = tickers[0]
        
        # 1. Build Structural DIBs on the Anchor Ticker (Threshold scaled to $50M)
        dibs = construct_m1_dibs(anchor, threshold=50_000_000)
        if dibs.empty: continue
            
        # 2. Load fast 1-min time bars for the whole basket to calculate continuous spread
        prices = {}
        for t in tickers:
            path = f"/Volumes/Vault/quant_data/tick data storage/{t}/parquet"
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.parquet') and not f.startswith('._')]
            if not files: continue
            df_t = pd.concat([pd.read_parquet(f, columns=['timestamp', 'price']) for f in files])
            df_t['timestamp'] = pd.to_datetime(df_t['timestamp'], utc=True)
            prices[t] = df_t.set_index('timestamp')['price'].resample('1min').last().ffill()

        if len(prices) != len(tickers): continue
        
        # 3. Calculate Spread & Sample at structural DIB timestamps
        spread_continuous = sum(prices[t] * weights[t] for t in tickers).dropna()
        spread_continuous = spread_continuous[~spread_continuous.index.duplicated(keep='last')]
        
        # Drop duplicates from DIBs to ensure perfectly unique structural timestamps
        unique_dib_index = dibs.index.drop_duplicates(keep='last')
        spread_dibs = spread_continuous.reindex(unique_dib_index, method='ffill').dropna()
        
        if len(spread_dibs) < 100: continue

        # 4. Feature Extraction: Dynamic Volatility & Fractional Differentiation
        vol = get_daily_vol(spread_dibs)
        opt_d, spread_fd = find_optimal_d(spread_dibs)
        print(f"  >> Stationarity achieved at d={opt_d:.2f}")
        
        # 5. Primary Model: Strict Z-Score Signals
        z = (spread_dibs - spread_dibs.rolling(50).mean()) / spread_dibs.rolling(50).std()
        events = z[(z > 2.5) | (z < -2.5)].to_frame('z')
        events['trgt'] = vol.reindex(events.index).ffill()
        events = events.dropna()
        
        if events.empty: continue

        # 6. Meta-Labeling: Triple Barrier Application
        labels = apply_triple_barrier(spread_dibs, events, pt_sl=[1, 2], t1=120)
        print(f"  >> Primary Signal Win Rate: {(labels['bin'].mean() * 100):.1f}%")
        
        # 7. Offline Microstructure Features (Instant calculation)
        print(f"  >> Calculating Local Microstructure Dynamics...")
        # Pass the anchor ticker's local price DataFrame
        df_anchor = prices[anchor].to_frame('price')
        df_anchor['size'] = 100 # Mocking size proxy for aggregated time bars
        micro = get_offline_microstructure(df_anchor)
        
        # 8. Assemble Feature Space Matrix
        X = pd.DataFrame(index=labels.index)
        X['frac_diff'] = spread_fd.reindex(labels.index).ffill()
        X['volatility'] = vol.reindex(labels.index).ffill()
        X['signal_strength'] = events['z']
        
        if not micro.empty:
            X['order_flow_imbalance'] = micro['order_flow_imbalance'].reindex(labels.index).ffill().fillna(0)
            X['effective_spread'] = micro['effective_spread'].reindex(labels.index).ffill().fillna(0)
            
        # SPACE FOR FUTURE FEATURES: (e.g., Options Implied Vol, NLP Sentiment)
        # X['future_feature_1'] = ...
        
        all_X.append(X.dropna())
        all_y.append(labels['bin'].reindex(X.dropna().index))

    if not all_X: 
        print("[WARNING] Insufficient data to train XGBoost.")
        return

    X_train = pd.concat(all_X)
    y_train = pd.concat(all_y)

    print(f"\nTraining Meta-Labeler on {len(X_train)} structural setups...")
    
    # Calculate exactly how imbalanced the dataset is
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    imbalance_ratio = float(num_neg) / num_pos if num_pos > 0 else 1.0
    print(f"  >> Auto-Balancing Weights (Ratio: {imbalance_ratio:.2f})")
    
    # 9. Hyperparameter Tuning with Purged CV
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        scale_pos_weight=imbalance_ratio, # Forces XGBoost to respect the minority wins
        n_jobs=-1,  
        random_state=42
    )
    
    cv_splits = custom_purged_kfold(pd.Series(X_train.index), n_splits=3)
    param_grid = {
        'max_depth': [2, 3, 5],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    
    search = RandomizedSearchCV(base_model, param_grid, cv=cv_splits, scoring='roc_auc', n_iter=10, verbose=0)
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    roc_score = search.best_score_
    print(f"[SUCCESS] Model Optimized. Cross-Validated ROC-AUC: {roc_score:.4f}")
    
    if roc_score < 0.55:
        print("  -> NOTE: ROC-AUC is relatively low. The model needs more historical data or additional alpha features (e.g., Order Flow Imbalance).")

    # 10. Versioning & Export
    v_files = glob.glob(os.path.join(MODELS_DIR, "meta_labeler_v*.json"))
    version = len(v_files) + 1
    model_name = f"meta_labeler_v{version}.json"
    model_path = os.path.join(MODELS_DIR, model_name)
    
    best_model.save_model(model_path)
    print(f"[SAVED] {model_name} securely exported to {MODELS_DIR}/")
    
    # Dynamically extract the features directly from the XGBoost memory
    used_features = best_model.feature_names
    feature_str = ", ".join(used_features) if used_features else "Unknown"
    
    version_file = os.path.join(MODELS_DIR, "active_model_version.txt")
    
    # 'a' mode appends the new model to the bottom of the ledger automatically
    with open(version_file, "a") as f:
        f.write(f"Model: {model_name} | ROC-AUC: {roc_score:.4f} | Features: [{feature_str}]\n")

if __name__ == "__main__":
    train_meta_labeler()