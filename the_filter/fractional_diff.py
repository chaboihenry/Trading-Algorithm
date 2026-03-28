import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from the_filter.dib_constructor import construct_dibs

def get_weights_ffd(d: float, threhsold: float = 1e-5):
    # fixed-width window fractional differentiation
    # stop when weight falls below the threhsold

    weights = [1.0]
    k = 1

    while True:
        next_weight = -weights[-1] * (d - k + 1) / k
        # if abs value of weight is smaller than threshold, stop
        if abs(next_weight) < threhsold:
            break
        
        weights.append(next_weight)
        k += 1
    
    return np.array(weights).reshape(-1, 1)

def frac_diff_ffd(series: pd.Series, d: float, threshold: float = 1e-5): 
    # compute weights for given d and threshold
    weights = get_weights_ffd(d, threshold)
    # format input series into dataframe
    df = series.to_frame('close')
    # reverse the weights array so most recent price gets 1.0 weight
    weights = weights[::-1]

    def dot_product(x):
        return np.dot(x, weights)[0]
    
    # apply rolling dot product across the price series
    df['frac_diff'] = df['close'].rolling(window=len(weights)).apply(dot_product, raw=True)

    return df

def find_min_d(series: pd.Series, min_d: float = 0.2):
    # search for the lowest d value that passes the adf test
    out = pd.DataFrame(columns=['adf_stat', 'p_value', 'lags', 'n_obs', '95%_conf', 'corr'])
    
    # enforce a minimum memory transformation to prevent tree-based models from overfitting
    num_steps = int(np.round((1.0 - min_d) * 10)) + 1
    d_values = np.linspace(min_d, 1.0, num_steps)
    
    for d in d_values:
        # calculate the fractionally differentiated series
        df_diff = frac_diff_ffd(series, d, threshold=1e-5)
        # drop the NaN rows created by the rolling window
        df_diff = df_diff.dropna()
        
        if df_diff.empty:
            continue
            
        # let the adf test dynamically find the optimal lags using aic
        adf_test = adfuller(df_diff['frac_diff'], regression='c', autolag='AIC')
        # record the p-value
        p_val = adf_test[1]
        out.loc[d] = [adf_test[0], p_val, adf_test[2], adf_test[3], adf_test[4]['5%'], 1.0]
        print(f"Testing d={d:.1f} | p-value: {p_val:.4f} | Lags Used: {adf_test[2]}")
        
        # if p-value is less than 0.05, it is statistically stationary
        if p_val < 0.05:
            print(f"\n[SUCCESS] Optimal d found: {d:.1f}")
            break
            
    return out


if __name__ == "__main__":
    test_path = 'data/tick data storage/V/parquet/ticks.parquet'
    print("==== PIPELINE INITIATED ====")
    # 1. construct dollar imbalance bars
    dib_df = construct_dibs(test_path, threshold=50_000_000)
    print("\n==== FINDING OPTIMAL MEMORY RETENTION ====")
    # 2. run the ADF stationarity search on closing prices of the DIBs
    results = find_min_d(dib_df['close'], min_d = 0.2)

