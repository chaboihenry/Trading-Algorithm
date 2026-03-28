import cudf 
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from typing import TypedDict

class PrisitineBasket(TypedDict):
    basket: list[str]
    half_life: float
    weights: dict[str, float]

# a sample pristine basket exported from cluster_discovery.py
sample_basket = {
    'basket': ['V', 'MA'],
    'half_life': 0.65, # trading days
    'weights': {'V': 1.0000, 'MA': -0.5412}
}

def generate_signals(portfolio_data: PrisitineBasket):
    tickers = portfolio_data['basket']
    half_life_days = portfolio_data['half_life']
    weights = portfolio_data['weights']

    print(f"\n==== GENERATING SIGNALS FOR BASKET: {tickers} ====")

    cpu_dataframes = []

    # load and resample the tick data (same as in cluster_discovery.py)
    for ticker in tickers:
        path = f"/app/data/tick data storage/{ticker}/parquet/ticks.parquet"
        arrow_table = pq.read_table(path, columns=['timestamp', 'price'])

        gdf = cudf.DataFrame.from_arrow(arrow_table).reset_index(drop=True)
        gdf['timestamp'] = cudf.to_datetime(gdf['timestamp'])
        gdf = gdf.sort_values('timestamp').set_index('timestamp')
        close_gpu = gdf['price'].resample('5min').last()

        df_cpu = close_gpu.reset_index().to_pandas()
        df_cpu.columns = ['timestamp', ticker]
        cpu_dataframes.append(df_cpu)
    
    print("Mergining pricing data...")
    aligned_data = cpu_dataframes[0]
    for df in cpu_dataframes[1:]:
        aligned_data = pd.merge(aligned_data, df, on='timestamp', how='inner')
    aligned_data = aligned_data.dropna().set_index('timestamp')

    print("Calculating synthetic spread using Johansen hedge ratios...")
    # spread = (Price_A * Weight_A) + (Price_B * Weight_B) + ...
    aligned_data['Spread'] = 0
    for ticker in tickers: 
        aligned_data['Spread'] += aligned_data[ticker] * weights[ticker]
    
    print("Computing dynamic Z-Score based on Ornstein-Uhlenbeck Half-Life...")
    # convert half-life days to 5-minute bars (78 bars per 6.5 trading day)
    # using 2x half-life window to capture the full mean-reversion cycle
    window_size = max(int((half_life_days * 78) * 2), 5)
    aligned_data['Rolling_Mean'] = aligned_data['Spread'].rolling(window=window_size).mean()
    aligned_data['Rolling_Std'] = aligned_data['Spread'].rolling(window=window_size).std()
    # calculate the z-score, dropping the initial NaN warmup period
    aligned_data['Z_Score'] = (aligned_data['Spread'] - aligned_data['Rolling_Mean']) / aligned_data['Rolling_Std']

    print("Executing threshold logid to generate target positions...")
    # +1 = LONG the spread (Buy V, Short MA)
    # -1 = SHORT the spread (Short V, Buy MA)
    # 0 = FLAT (liquidate to cash)
    aligned_data['Target_Position'] = np.nan
    # set entry signals based on extreme standard deviations
    aligned_data.loc[aligned_data['Z_Score'] < -2.0, 'Target_Position'] = 1
    aligned_data.loc[aligned_data['Z_Score'] > 2.0, 'Target_Position'] = -1
    # set exit signals (0) when the leash snaps back across the mean
    aligned_data.loc[(aligned_data['Z_Score'] >= 0) & (aligned_data['Z_Score'].shift(1) < 0), 'Target_Position'] = 0
    aligned_data.loc[(aligned_data['Z_Score'] <= 0) & (aligned_data['Z_Score'].shift(1) > 0), 'Target_Position'] = 0
    # forward-fill the active positions, and fill any leading/empty periods with 0 (flat)
    aligned_data['Target_Position'] = aligned_data['Target_Position'].ffill().fillna(0)
    # count how many times the target position physically changed state
    total_trades = (aligned_data['Target_Position'].diff().abs() > 0).sum()
    print(f"Generated {total_trades} historical state changes (Entries/Exits).")
    
    return aligned_data


if __name__ == "__main__":
    signal_df = generate_signals(sample_basket)
    print("\n===== RECENT SIGNALS ====")
    print(signal_df[['Spread', 'Z_Score', 'Target_Position']].tail(15))