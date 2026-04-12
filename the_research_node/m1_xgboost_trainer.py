import os
import json
import glob
import pandas as pd
import numpy as np
import xgboost as xgb
from numba import jit
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import RandomizedSearchCV

# --- CONFIGURATION & ENV ---
UNIVERSE_PATH = "the_models/curated_universe.json"
MODELS_DIR = "the_models"

# --- 1. DIB CONSTRUCTION ENGINE ---
@jit(nopython=True)
def sample_imbalance_bars_streaming(signed_dv_array, threshold, initial_theta):
    bar_indices = np.empty(len(signed_dv_array), dtype=np.int64)
    count = 0
    theta = initial_theta

    for i in range(len(signed_dv_array)):
        theta += signed_dv_array[i]
        if abs(theta) >= threshold:
            bar_indices[count] = i
            count += 1
            theta = 0.0 # Path-dependent reset
    
    return bar_indices[:count], theta

def construct_m1_dibs(ticker: str, threshold: float = 50_000_000):
    print(f"Constructing Streaming DIBs for Anchor Ticker: {ticker}...")
    path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"
    
    # 1. Sort files chronologically and strictly enforce the 2024+ regime
    try:
        files = sorted([
            os.path.join(path, f) for f in os.listdir(path) 
            if f.endswith('.parquet') and not f.startswith('._') and f >= "2024_01.parquet"
        ])
    except FileNotFoundError:
        return pd.DataFrame()
        
    if not files: return pd.DataFrame()
    
    dibs_list = []
    leftover_theta = 0.0
    
    for f in files:
        try:
            df = pd.read_parquet(f, columns=['timestamp', 'price', 'size'])
            if df.empty: continue
            
            df['dollar_volume'] = df['price'] * df['size']
            
            # Stable tick direction
            diff = df['price'].diff()
            tick_dir = np.sign(diff)
            tick_dir.iloc[0] = 1.0 # Default first tick
            df['tick_direction'] = tick_dir.replace(0, np.nan).ffill().fillna(1)
            
            df['signed_dollar_volume'] = df['dollar_volume'] * df['tick_direction']
            signed_dv_np = df['signed_dollar_volume'].values.astype(np.float64)
            
            # Pass the running theta balance between monthly files
            sampled_indices, leftover_theta = sample_imbalance_bars_streaming(
                signed_dv_np, threshold, leftover_theta
            )
            
            if len(sampled_indices) > 0:
                chunk_dibs = df.iloc[sampled_indices][['timestamp', 'price']].rename(columns={'price': 'close'})
                dibs_list.append(chunk_dibs)
                
            # Force memory release of the huge raw arrays
            del df, signed_dv_np, diff, tick_dir
            
        except Exception:
            continue
            
    if not dibs_list: return pd.DataFrame()
    
    # Combine the tiny sampled chunks (Kilobytes of RAM)
    dibs = pd.concat(dibs_list, ignore_index=True)
    dibs = dibs.set_index('timestamp')
    
    return dibs

# --- 2. FRACTIONAL DIFFERENTIATION (Memory Preservation Engine) ---
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

# --- 3. TRIPLE BARRIER LABELING ENGINE ---
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

# --- 4. PURGED K-FOLD CV (Prevents Data Leakage) ---
def custom_purged_kfold(times: pd.Series, n_splits: int = 3, embargo_pct: float = 0.01, t1_hours: int = 120):
    # Purge both sides of the test fold by the barrier horizon 
    indices = np.arange(len(times))
    test_splits = np.array_split(indices, n_splits)
    
    # Embargo sized to the label horizon, not an arbitrary %
    embargo_length = max(
        int(len(times) * embargo_pct),
        t1_hours  # minimum purge = barrier width
    )
    
    splits = []
    for test_idx in test_splits:
        test_start = test_idx[0]
        test_end = test_idx[-1]
        
        # Purge before test: any train sample whose barrier window could leak into the test period
        purge_before = max(0, test_start - embargo_length)
        
        # Purge AFTER test: original embargo
        purge_after = min(len(indices), test_end + embargo_length + 1)
        
        train_idx = np.concatenate([
            indices[:purge_before],
            indices[purge_after:]
        ])
        
        splits.append((train_idx, test_idx))
    return splits

# --- 5. MAIN META-LABELER PIPELINE ---
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
    total_baskets = len(baskets)

    for idx, (name, data) in enumerate(baskets.items(), 1):
        # native terminal progress bar
        pct = (idx / total_baskets) * 100
        filled = int(30 * idx // total_baskets)
        bar = '█' * filled + '-' * (30 - filled)
        print(f"\n[{idx}/{total_baskets}] {name} |{bar}| {pct:.1f}%")
        
        tickers = data['tickers']
        weights = data['weights']
        anchor = tickers[0]
        
        # 1. Build Structural DIBs on the Anchor Ticker (Threshold scaled to $50M)
        dibs = construct_m1_dibs(anchor, threshold=50_000_000)
        if dibs.empty: continue
            
        # 2. Stream and instantly resample 1-min bars chronologically
        prices = {}
        for t in tickers:
            path = f"/Volumes/Vault/quant_data/tick data storage/{t}/parquet/training_data"
            try:
                files = sorted([
                    os.path.join(path, f) for f in os.listdir(path) 
                    if f.endswith('.parquet') and not f.startswith('._') and f >= "2024_01.parquet"
                ])
            except FileNotFoundError:
                continue
                
            if not files: continue
            
            chunk_resampled = []
            for f in files:
                try:
                    df_t = pd.read_parquet(f, columns=['timestamp', 'price'])
                    if df_t.empty: continue
                    # Add .shift(1) to prevent the Pandas right-edge lookahead bias
                    df_t = df_t.set_index('timestamp')['price'].resample('1min').last().ffill().shift(1)
                    chunk_resampled.append(df_t)
                except Exception:
                    continue
                    
            if chunk_resampled:
                prices[t] = pd.concat(chunk_resampled)

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
        
        # 7. Assemble Feature Space Matrix
        X = pd.DataFrame(index=labels.index)
        X['frac_diff'] = spread_fd.reindex(labels.index).ffill()
        X['volatility'] = vol.reindex(labels.index).ffill()
        X['signal_strength'] = events['z']
            
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
    
    # 8. Hyperparameter Tuning with Purged CV
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

    # 9. Versioning & Export
    v_files = glob.glob(os.path.join(MODELS_DIR, "meta_labeler_v*.json"))
    version = len(v_files) + 1
    model_name = f"meta_labeler_v{version}.json"
    model_path = os.path.join(MODELS_DIR, model_name)
    
    best_model.save_model(model_path)
    print(f"[SAVED] {model_name} securely exported to {MODELS_DIR}/")
    
    # Dynamically extract the features directly from the XGBoost memory
    used_features = best_model.feature_names_in_
    feature_str = ", ".join(used_features) if used_features is not None and len(used_features) > 0 else "Unknown"
    
    version_file = os.path.join(MODELS_DIR, "active_model_version.txt")

    # Overwrite with only the current active model
    with open(version_file, "w") as f:
        f.write(f"Model: {model_name} | ROC-AUC: {roc_score:.4f} | Features: [{feature_str}]\n")

if __name__ == "__main__":
    train_meta_labeler()