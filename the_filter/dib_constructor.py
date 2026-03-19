import cudf
import numpy as np
from numba import jit
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import gc

# compile the path-dependent loop into C-code for extreme speed
@jit(nopython=True)
def sample_imbalance_bars(signed_dv_array, threshold):
    # pre-allocate the maximum possible array size. this prevents dynamic resizing crashes.
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

def construct_dibs(parquet_path: str, threshold: float = 50_000_000):
    print(f"\n1. Loading raw tick data from {parquet_path}...")
    # read the raw file via pyarrow to intercept the timezone tag
    arrow_table = pq.read_table(parquet_path, columns = ['timestamp', 'price', 'size'])
    
    # forcefully strip the timezone tag so cuDF doesn't crash during sort
    naive_timestamp = pc.cast(arrow_table['timestamp'], pa.timestamp('ns'))
    timestamp_idx = arrow_table.column_names.index('timestamp')
    arrow_table = arrow_table.set_column(timestamp_idx, 'timestamp', naive_timestamp)

    # transfer sanitized data to GPU VRAM
    gdf = cudf.DataFrame.from_arrow(arrow_table).reset_index(drop=True)
    gdf = gdf.rename(columns={'size': 'volume'})

    # Incinerate protocol, nuke massive pyarrow table from CPU RAM to prevent Docker OOM SegFault
    del arrow_table
    gc.collect()
    
    # because the timezone is gone, cuDF can sort natively on the RTX 5080
    gdf = gdf.sort_values('timestamp').reset_index(drop=True)

    print("2. Calculating tick direction and signed dollar volume...")
    gdf['dollar_volume'] = gdf['price'] * gdf['volume']
    gdf['price_change'] = gdf['price'].diff()
    gdf['tick_direction'] = 0
    gdf.loc[gdf['price_change'] > 0, 'tick_direction'] = 1
    gdf.loc[gdf['price_change'] < 0, 'tick_direction'] = -1
    gdf['tick_direction'] = gdf['tick_direction'].replace(0, cudf.NA).ffill().fillna(1)
    gdf['signed_dollar_volume'] = gdf['dollar_volume'] * gdf['tick_direction']

    print("3. Safely bridging single target array to CPU...")
    # bypass cuDF's bloated transfer logic entirely.
    # extract the naked gpu array and use cupy for a direct PCIe DMA transfer.
    signed_dv_np = gdf['signed_dollar_volume'].fillna(0).astype('float32').to_arrow().to_numpy()

    print(f"4. Unleashing C-compiled sampling loop (Threshold: ${threshold:,.0f} imbalance)...")
    # process millions of rows in milliseconds (hopefully)
    sampled_indices = sample_imbalance_bars(signed_dv_np, threshold)
    print(f"   -> Extracted {len(sampled_indices)} Dollar Imbalance Bars.")

    print("5. Constructing final OHLCV Microstructure features on GPU VRAM...")
    # push the winning indices back to the GPU using CuPy 
    group_flags = np.zeros(len(gdf), dtype=np.int32)
    group_flags[sampled_indices] = 1
    group_ids = np.cumsum(group_flags)
    group_ids = np.roll(group_ids, 1)
    group_ids[0] = 0
    gdf['group_id'] = group_ids

    # aggregate the raw ticks into the new DIBs
    dibs_gpu = gdf.groupby('group_id').agg({
        'timestamp': 'last', 
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum', 
        'dollar_volume': 'sum'
    })

    # flatten multi-index columns natively in cuDF
    dibs_gpu.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'dollar_volume']
    dibs_gpu = dibs_gpu.set_index('timestamp')
    
    print("6. Executing Final Sweep to clear VRAM and CPU memory...")
    # violently delete the 30-million row variables so Docker has RAM for the Pandas conversion
    del gdf
    del signed_dv_np
    del group_flags
    del group_ids
    gc.collect()
    
    print("7. Transferring pristine execution bars to CPU...")
    return dibs_gpu.to_arrow().to_pandas()

if __name__ == "__main__":
    # test on Visa. set the imbalance threshold to $50 Million
    # a bar will only print when buyers outpace sellers (or vice versa) by $50M.
    test_path = '/app/data/tick data storage/V/parquet/ticks.parquet'
    dib_df = construct_dibs(test_path, threshold=50_000_000)
    print("\n=== PRISTINE DOLLAR IMBALANCE BARS (XGBOOST FEATURES) ===")
    print(dib_df.tail(10))