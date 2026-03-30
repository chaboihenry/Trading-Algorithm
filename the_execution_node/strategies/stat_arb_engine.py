# the_execution_node/strategies/stat_arb_engine.py

import os
import json
import pandas as pd
import numpy as np

def generate_signals(live_matrix: pd.DataFrame, models_dir: str = "the_models"):
    # Ingests the live RAM matrix and calculates current Z-Scores based on the M1 payload.
    # Outputs active trading signals and historical spread returns.
    
    # 1. Load the dynamic curated universe from the JSON payload
    path = os.path.join(models_dir, "curated_universe.json")
    try:
        with open(path, "r") as f:
            data = json.load(f)
            curated_baskets = data.get("baskets", {})
    except FileNotFoundError:
        print("[CRITICAL] curated_universe.json not found in stat_arb_engine.")
        return {}, pd.DataFrame()

    spread_series = {}
    active_signals = {}

    for spread_name, params in curated_baskets.items():
        weights = params.get('weights', {})
        half_life = params.get('half_life', 1.0)
        allocation = params.get('capital_allocation', 0.0)

        # Skip quarantined spreads assigned 0.00% by HRP
        if allocation <= 0:
            continue

        # 2. Calculate the synthetic spread price over the live window
        spread_val = pd.Series(0.0, index=live_matrix.index)
        valid_legs = True
        
        for ticker, w in weights.items():
            if ticker in live_matrix.columns:
                spread_val += live_matrix[ticker] * w
            else:
                valid_legs = False
                break
        
        if not valid_legs or spread_val.empty:
            continue

        spread_series[spread_name] = spread_val

        # 3. Compute dynamic Z-Score mapping days to 1-minute bars (390 mins/day)
        window = max(int(half_life * 390), 30) # Enforce a minimum 30-minute window
        
        # We need enough data points in the live matrix to calculate a valid rolling metric
        if len(spread_val) < window:
            continue

        rolling_mean = spread_val.rolling(window=window).mean()
        rolling_std = spread_val.rolling(window=window).std()
        
        # Prevent division by zero if the spread goes flat
        if rolling_std.iloc[-1] == 0 or pd.isna(rolling_std.iloc[-1]):
            continue
            
        z_score = (spread_val - rolling_mean) / rolling_std

        # 4. Isolate the CURRENT moment in time (the last row)
        current_z = z_score.iloc[-1]

        # 5. Generate Target Position based on standard deviations
        target_pos = 0
        if current_z < -2.0:
            target_pos = 1  # Spread is undervalued -> Long
        elif current_z > 2.0:
            target_pos = -1 # Spread is overvalued -> Short

        if target_pos != 0:
            active_signals[spread_name] = {
                'johansen_weights': weights,
                'target_position': target_pos,
                'hrp_allocation': allocation,
                'current_z': current_z
            }

    # Compile the raw spread prices into a DataFrame
    spread_data = pd.DataFrame(spread_series)

    # 6. Spreads cross zero, so use absolute differences
    if spread_data.empty:
        spread_returns = pd.DataFrame()
    else:
        spread_returns = spread_data.diff().dropna()

    return active_signals, spread_returns


if __name__ == "__main__":
    print("\n====== EXECUTION NODE DIAGNOSTIC: STAT ARB ENGINE ======")
    
    # 1. Create a synthetic live matrix with 100 rows (minutes) of dummy data
    print("[SYSTEM] Generating synthetic live matrix (100 ticks)...")
    np.random.seed(42)
    dummy_tickers = ['AAPL', 'MSFT', 'V', 'MA', 'GOOG', 'GOOGL']
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='1min')
    
    # Simulate random walks for prices
    prices = np.random.randn(100, len(dummy_tickers)).cumsum(axis=0) + 100
    live_matrix = pd.DataFrame(prices, index=dates, columns=dummy_tickers)
    
    # 2. To test the logic without waiting for 390 minutes, we temporarily override the JSON
    # by passing a fake models directory (which will fail gracefully), so I can inject a test basket.
    print("[SYSTEM] Running Engine evaluation...")
    
    # Injecting a flash crash shock
    # AAPL bounces naturally around 150 for 99 mins, creating a tight standard deviation.
    # Then it gaps down violently to 135 on the very last minute.
    noise = np.random.normal(0, 0.2, 99) 
    live_matrix['AAPL'] = np.concatenate([150 + noise, [135]])
    live_matrix['MSFT'] = np.full(100, 300) # MSFT stays completely flat
    
    # Manually creating the payload that the JSON usually provides
    test_baskets = {
        'AAPL_MSFT_Spread': {
            'weights': {'AAPL': 1.0, 'MSFT': -0.5},
            'half_life': 0.1, # Short half-life for testing (39 min window)
            'capital_allocation': 0.15
        }
    }
    
    # Temporarily patch the json.load just for this test block
    import builtins
    original_open = builtins.open
    
    class MockFile:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps({"baskets": test_baskets})
        
    def mock_open(*args, **kwargs):
        if "curated_universe.json" in args[0]: return MockFile()
        return original_open(*args, **kwargs)
        
    builtins.open = mock_open
    
    # 3. Execute the Engine
    signals, returns = generate_signals(live_matrix, models_dir=".")
    
    print("\n====== DIAGNOSTIC RESULTS ======")
    if not signals:
        print("[WARNING] No signals generated. Spread did not cross Z-Score thresholds.")
    else:
        for spread, data in signals.items():
            print(f"[ACTION] Signal Fired for {spread}:")
            print(f"  -> Target Position: {'LONG' if data['target_position'] == 1 else 'SHORT'}")
            print(f"  -> Allocation: {data['hrp_allocation'] * 100}%")
            print(f"  -> Z-Score: {data['current_z']:.2f}")
            
    print(f"\n[SUCCESS] Spread Returns Shape: {returns.shape}")
    
    # Restore original open function
    builtins.open = original_open