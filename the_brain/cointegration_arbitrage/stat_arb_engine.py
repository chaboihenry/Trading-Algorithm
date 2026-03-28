import pandas as pd
import numpy as np

def generate_signals(live_matrix: pd.DataFrame):
    """
    Ingests the live RAM matrix from The Oracle and calculates current Z-Scores.
    Outputs the active trading signals and the historical spread returns.
    """
    # in a fully deployed setup, these come from the M1's curated_universe.json
    # defined here to process the live matrix.
    curated_baskets = {
        'V_MA_Spread': {'weights': {'V': 1.0, 'MA': -0.5412}, 'half_life': 12},
        'AAPL_MSFT_Spread': {'weights': {'AAPL': 1.0, 'MSFT': -0.8921}, 'half_life': 15}
    }
    
    spread_series = {}
    active_signals = {}
    
    for spread_name, params in curated_baskets.items():
        weights = params['weights']
        
        # 1. Calculate the synthetic spread price over the 7-day window
        spread_val = pd.Series(0.0, index=live_matrix.index)
        for ticker, w in weights.items():
            if ticker in live_matrix.columns:
                spread_val += live_matrix[ticker] * w
                
        spread_series[spread_name] = spread_val
        
        # 2. Compute dynamic Z-Score using the OU Half-Life
        window = int(params['half_life'] * 2) 
        rolling_mean = spread_val.rolling(window=window).mean()
        rolling_std = spread_val.rolling(window=window).std()
        z_score = (spread_val - rolling_mean) / rolling_std
        
        # 3. Isolate the CURRENT moment in time (the last row)
        current_z = z_score.iloc[-1]
        
        # 4. Generate Target Position
        target_pos = 0
        if current_z < -2.0:
            target_pos = 1  # Long the Spread
        elif current_z > 2.0:
            target_pos = -1 # Short the Spread
            
        if target_pos != 0:
            active_signals[spread_name] = {
                'johansen_weights': weights,
                'target_position': target_pos
            }
            
    # Compile the raw spread prices into a DataFrame
    spread_data = pd.DataFrame(spread_series)
    
    # The Allocator and Filter need Returns, not absolute prices
    spread_returns = np.log(spread_data / spread_data.shift(1)).dropna()
    
    return active_signals, spread_returns