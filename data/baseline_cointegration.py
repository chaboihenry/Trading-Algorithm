import cudf
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.tsa.stattools as ts

def test_baseline_cointegration(path_a: str, path_b: str):
    # read parquet metadata and stitch chunks using pyarrow to bypass memory bug
    print("Reading parquet metadata and chunking via pyarrow on CPU...")
    arrow_a = pq.read_table(path_a, columns=['timestamp', 'price'])
    arrow_b = pq.read_table(path_b, columns=['timestamp', 'price'])

    # transfer unified arrays directly to GPU VRAM
    print("Transferring unified arrays to GPU VRAM...")
    gdf_a = cudf.DataFrame.from_arrow(arrow_a).reset_index(drop=True)
    gdf_b = cudf.DataFrame.from_arrow(arrow_b).reset_index(drop=True)
    print(f"Asset A loaded with {len(gdf_a)} ticks. Asset B loaded with {len(gdf_b)} ticks.")

    # explicitly cast to datetime and sort before setting the index
    print("Formatting timestamps and sorting...")
    gdf_a['timestamp'] = cudf.to_datetime(gdf_a['timestamp'])
    gdf_b['timestamp'] = cudf.to_datetime(gdf_b['timestamp'])
    gdf_a = gdf_a.sort_values('timestamp').set_index('timestamp')
    gdf_b = gdf_b.sort_values('timestamp').set_index('timestamp')

    # resample to 5-minute bars on CUDA cores
    print("Resampling 89+ million ticks to 5-minute bars on CUDA cores...")
    close_a = gdf_a['price'].resample('5min').last()
    close_b = gdf_b['price'].resample('5min').last()

    # flatten individual arrays on GPU to bypass cuDF join bug
    print("Flattening individual arrays on GPU to bypass cuDF join bug...")
    df_a_gpu = close_a.reset_index()
    df_a_gpu.columns = ['timestamp', 'Asset_A']
    df_b_gpu = close_b.reset_index()
    df_b_gpu.columns = ['timestamp', 'Asset_B']

    # transfer tiny, flattened 5-minute dataframes to CPU
    print("Transferring tiny, flattened 5-minute DataFrames to CPU...")
    df_a_cpu = df_a_gpu.to_pandas()
    df_b_cpu = df_b_gpu.to_pandas()

    # merge aligned data cleanly on the CPU
    print("Merging aligned data cleanly on the CPU...")
    aligned_data = pd.merge(df_a_cpu, df_b_cpu, on='timestamp', how='inner')
    aligned_data = aligned_data.dropna().set_index('timestamp')

    # failsafe
    if len(aligned_data) == 0:
        print("\nError: Aligned data is empty. The timestamps between the two files do not overlap.")
        return

    # run ADF cointegration test
    print(f"Running ADF Test on {len(aligned_data)} aligned 5-minute bars...")
    score, p_value, _ = ts.coint(aligned_data['Asset_A'], aligned_data['Asset_B'])

    print(f"Cointegration T-Statistic: {score}")
    print(f"P-Value: {p_value}")

    if p_value < 0.05:
        print("\nThe leash is real! The baseline pair is cointegrated")
        return p_value, True
    else:
        print("\nNo cointegration found. Check data alignment.")
        return p_value, False


if __name__ = "__main__":
    # local bucket paths for the baseline pair (GOOG vs GOOGL)
    local_goog = '/app/tick data storage/GOOG/parquet/ticks.parquet'
    local_googl = '/app/tick data storage/GOOGL/parquet/ticks.parquet'
    test_baseline_cointegration(local_goog, local_googl)