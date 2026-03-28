import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockQuotesRequest
from alpaca.data.timeframe import TimeFrame

from the_filter.purged_kfold import PurgedKFold

def get_alpaca_microstructure(symbol: str, start_dt, end_dt, api_key: str, secret_key: str):
    client = StockHistoricalDataClient(api_key, secret_key)
    request_params = StockQuotesRequest(symbol_or_symbols=symbol, start=start_dt, end=end_dt)
    
    print(f"Fetching Level 1 Quotes for {symbol}...")
    quotes = client.get_stock_quotes(request_params).df
    
    quotes = quotes[(quotes['ask_price'] > 0) & (quotes['bid_price'] > 0)]
    quotes = quotes[(quotes['ask_size'] > 0) & (quotes['bid_size'] > 0)]
    
    quotes['mid_price'] = (quotes['ask_price'] + quotes['bid_price']) / 2.0
    quotes['micro_price'] = (
        (quotes['bid_price'] * quotes['ask_size']) + (quotes['ask_price'] * quotes['bid_size'])
    ) / (quotes['bid_size'] + quotes['ask_size'])
    
    quotes['micro_price_divergence'] = quotes['micro_price'] - quotes['mid_price']
    quotes['relative_spread'] = (quotes['ask_price'] - quotes['bid_price']) / quotes['mid_price']
    
    micro_features = quotes[['micro_price_divergence', 'relative_spread']].resample('5min').mean().ffill()
    return micro_features

def prep_meta_labels(features_df: pd.DataFrame, labels_df: pd.DataFrame):
    # inner join ensures we only train on data where we have both a signal and an outcome
    df = features_df.join(labels_df, how='inner').dropna()
    
    # convert to binary classification
    df['meta_label'] = (df['bin'] == 1.0).astype(int)
    
    # ensure the 't1' (end of trade barrier time) is preserved for PurgedKFold
    if 'first_touch' in df.columns:
        df['end_time'] = df['first_touch']
    elif 't1' in df.columns:
        df['end_time'] = df['t1']
    else:
        raise ValueError("CRITICAL: Triple-Barrier end time ('t1' or 'first_touch') missing from labels.")
        
    return df

def train_and_optimize_model(df: pd.DataFrame):
    feature_cols = [
        'frac_diff', 'volatility', 'Z_Score', 'Target_Position',
        'micro_price_divergence', 'relative_spread'
    ]
    
    X = df[feature_cols]
    y = df['meta_label']
    times = df['end_time'] # info range for Purging
    
    # 1. Outer Split: strict Chronological Train/Test split (80/20)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    times_train = times.iloc[:split_idx]
    
    print(f"Optimizing on {len(X_train)} samples, Out-of-Sample Testing on {len(X_test)} samples...")
    
    # 2. Inner Split: instantiate AFML Purged K-Fold for the Training set
    # Using 3 splits and a 1% embargo to kill structural memory leakage
    pkf = PurgedKFold(n_splits=3, times=times_train, embargo=0.01)
    
    # convert the PKF generator into a list of tuples for Scikit-Learn
    cv_splits = list(pkf.split(X_train))
    
    # 3. Define the hyperparameter grid
    param_grid = {
        'max_depth': [2, 3, 4], # Keep trees shallow to prevent overfitting
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 200],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }
    
    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        device='cuda', # leverage RTX 5080 GPU
        random_state=42,
        eval_metric='auc'
    )
    
    # 4. Randomized Search with Purged K-Fold
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=15,          
        scoring='roc_auc',  
        cv=cv_splits,       # leak-proof CV splits
        verbose=1,
        random_state=42
    )
    
    print("Unleashing CUDA cores for Purged Hyperparameter Tuning...")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"\n[SUCCESS] Optimal Parameters Found: {search.best_params_}")
    
    # 5. Evaluate on the untouched 20% Out-Of-Sample Test Set
    preds = best_model.predict(X_test)
    probs = best_model.predict_proba(X_test)[:, 1]
    
    print("\n=== OPTIMIZED XGBOOST METRICS ===")
    print(classification_report(y_test, preds))
    print(f"Test ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # 6. Save the model for Jupyter analysis
    os.makedirs('model', exist_ok=True)
    model_path = 'model/meta_labeler_v1.json'
    best_model.save_model(model_path)
    print(f"\n[SAVED] Model serialized to: {model_path}")
    
    results = X_test.copy()
    results['Actual_Outcome'] = y_test
    results['Probability_of_Success'] = probs
    
    return best_model, results

def calculate_kelly_size(prob: float, payoff_ratio: float = 2.0):
    if prob < 0.50:
        return 0.0 
    kelly_fraction = prob - ((1 - prob) / payoff_ratio)
    return max(0.0, kelly_fraction / 2.0)


if __name__ == "__main__":
    print("\n=== INITIALIZING SECURE ENVIRONMENT ===")
    load_dotenv()
    
    API_KEY = os.getenv("ALPACA_API_KEY")
    SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
    
    if not API_KEY or not SECRET_KEY:
        raise ValueError("CRITICAL: Alpaca API keys not found. Ensure your .env file is configured correctly.")
        
    print("[SUCCESS] API Keys loaded securely from environment.")
    print("[Architecture Built with Purged K-Fold. Waiting for Live Dataframes to execute.]")

