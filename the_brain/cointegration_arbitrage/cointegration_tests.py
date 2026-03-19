import cudf
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

def test_johansen_cointegration(paths: list):
    print(f"\nEvaluating Johansen Cointegration for {len(paths)} assets...")
    
    cpu_dataframes = []
    tickers_in_basket = []
    
    # loop through all provided paths, resample on gpu, and prep for merge
    for path in paths:
        ticker = path.split('/')[3]  # extract ticker from path
        tickers_in_basket.append(ticker)
        
        print(f"Reading and chunking {ticker} via pyarrow...")
        arrow_table = pq.read_table(path, columns=['timestamp', 'price'])
    
        print(f"Transferring {ticker} to GPU VRAM and formatting...")
        gdf = cudf.DataFrame.from_arrow(arrow_table).reset_index(drop=True)
        gdf['timestamp'] = cudf.to_datetime(gdf['timestamp'])
        gdf = gdf.sort_values('timestamp').set_index('timestamp')
        
        print(f"Resampling {ticker} ticks to 5-minute bars on CUDA cores...")
        close_gpu = gdf['price'].resample('5min').last()
        
        # flatten and transfer back to cpu to avoid join memory bugs
        df_cpu = close_gpu.reset_index().to_pandas()
        df_cpu.columns = ['timestamp', ticker]
        cpu_dataframes.append(df_cpu)

    # dynamically merge all dataframes in the basket on the timestamp
    print("\nMerging all aligned assets cleanly on the CPU...")
    aligned_data = cpu_dataframes[0]
    for df in cpu_dataframes[1:]:
        aligned_data = pd.merge(aligned_data, df, on='timestamp', how='inner')
    
    aligned_data = aligned_data.dropna().set_index('timestamp')
    
    if len(aligned_data) == 0:
        print("\nError: Aligned data is empty. The timestamps do not overlap.")
        return False, None, None

    print(f"Running Johansen Test on {len(aligned_data)} aligned 5-minute bars...")
    # det_order=0 (no deterministic trend), k_ar_diff=1 (1 lag in VAR)
    johansen_result = coint_johansen(aligned_data, det_order=0, k_ar_diff=1)
    
    # the trace statistic tests the null hypothesis of 0 cointegrating relationships.
    # index 0 is the trace stat for r=0. index 1 of cvt is the 95% critical value.
    trace_stat = johansen_result.lr1[0]
    crit_value_95 = johansen_result.cvt[0, 1]
    
    print(f"\nTrace Statistic: {trace_stat}")
    print(f"95% Critical Value: {crit_value_95}")
    
    if trace_stat > crit_value_95:
        print("\nThe leash is real! The basket is cointegrated.")
        # extract winning eigenvector...
        eigenvector = johansen_result.evec[:, 0]
        hedge_ratios = eigenvector / eigenvector[0]
        weights = dict(zip(tickers_in_basket, hedge_ratios))
        spread = aligned_data.dot(eigenvector)
        # calc ornstein-uhlenbeck half-life...
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        spread_diff = spread_diff.loc[spread_lag.index]
        # delta_spread = lambda * lagged_spread
        X = sm.add_constant(spread_lag)
        ols_model = sm.OLS(spread_diff, X).fit()
        lambda_val = ols_model.params.iloc[1]
        # if lambda is positive, it's divergent (not mean reverting)
        if lambda_val > 0:
            half_life_bars = np.inf
        else:
            half_life_bars = -np.log(2) / lambda_val
        # convert 5-min bars to trading days (assuming 6.5 hour trading day = 78 bars)
        half_life_days = half_life_bars / 78

        return True, half_life_days, weights
    else:
        print("\nNo cointegration found in this basket.")
        return False, None, None

if __name__ == "__main__":
    # test the upgraded engine on the baseline pair
    local_goog = '/app/data/tick data storage/GOOG/parquet/ticks.parquet'
    local_googl = '/app/data/tick data storage/GOOGL/parquet/ticks.parquet'
    test_johansen_cointegration([local_goog, local_googl])