import os
import json
import pandas as pd

from the_utilities.paths import EXECUTION_DATA_DIR, STRUCTURAL_LIFECYCLE_JSON, BACKTEST_PARQUET


def compile_historical_state(lookback_years: int = 5):
    # Compiles 5-minute historical state for every ticker that ever cointegrated
    # across the 5-year lifecycle ledger (not just the current trading universe).

    print(f"\n====== COMPILING {lookback_years}-YEAR HISTORICAL STATE ======")

    # 1. Load the 5-year lifecycle ledger and extract the full ticker set
    try:
        with open(STRUCTURAL_LIFECYCLE_JSON, "r") as f:
            ledger = json.load(f)
    except FileNotFoundError:
        print(f"[CRITICAL] {STRUCTURAL_LIFECYCLE_JSON} not found. Run m1_structural_profiler first.")
        return

    # Collect every ticker that ever appeared in any basket's lifecycle
    target_tickers = set()
    for basket_data in ledger.values():
        target_tickers.update(basket_data.get("tickers", []))
    target_tickers = sorted(target_tickers)

    print(f"[SYSTEM] Target Universe: {len(target_tickers)} unique tickers across {len(ledger)} historical baskets.")
    print(f"[SYSTEM] Tickers: {target_tickers}")

    # 2. Establish time constraints
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.DateOffset(years=lookback_years)
    cutoff_str = cutoff_dt.strftime('%Y%m%d')
    print(f"[SYSTEM] Extracting data from {cutoff_dt.strftime('%Y-%m-%d')} to Present...")

    ticker_series = {}
    missing_tickers = []

    # 3. Extract and resample from the Vault
    for ticker in target_tickers:
        path = f"/Volumes/Vault/quant_data/tick data storage/{ticker}/parquet/training_data"

        if not os.path.exists(path):
            print(f"  -> [WARNING] Vault path missing for {ticker}. Skipping.")
            missing_tickers.append(ticker)
            continue

        print(f"  -> Processing {ticker}...")

        resampled_chunks = []
        for file in sorted(os.listdir(path)):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
            if file[:8] < cutoff_str:
                continue

            file_path = os.path.join(path, file)
            try:
                chunk = pd.read_parquet(file_path, columns=['timestamp', 'price'])
                if chunk.empty:
                    continue
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
                bars = chunk.set_index('timestamp')['price'].resample('5min').last()
                resampled_chunks.append(bars)
                del chunk
            except Exception:
                continue

        if not resampled_chunks:
            missing_tickers.append(ticker)
            continue

        series = pd.concat(resampled_chunks).sort_index()
        ticker_series[ticker] = series[~series.index.duplicated(keep='last')].ffill()

    if not ticker_series:
        print("[CRITICAL] No valid data extracted from the vault.")
        return

    # 4. Merge all series into one matrix
    print("\n[SYSTEM] Merging all assets into a unified matrix...")
    master_matrix = pd.DataFrame(ticker_series)
    master_matrix = master_matrix.ffill().dropna()

    # 5. Export
    os.makedirs(EXECUTION_DATA_DIR, exist_ok=True)

    print(f"[SYSTEM] Writing to {BACKTEST_PARQUET}...")
    master_matrix.to_parquet(BACKTEST_PARQUET, engine='pyarrow', compression='snappy')

    file_size_mb = os.path.getsize(BACKTEST_PARQUET) / (1024 * 1024)

    print("\n====== EXPORT SUCCESS ======")
    print(f"Matrix Shape:    {master_matrix.shape[0]} rows x {master_matrix.shape[1]} columns")
    print(f"File Size:       {file_size_mb:.2f} MB")
    print(f"Missing Tickers: {missing_tickers if missing_tickers else 'None'}")

    # 6. Warn about baskets that will still be skipped due to missing tickers
    if missing_tickers:
        print("\n[WARNING] Baskets that will be skipped in backtest due to missing ticker data:")
        for spread_name, basket_data in ledger.items():
            missing_in_basket = [t for t in basket_data["tickers"] if t in missing_tickers]
            if missing_in_basket:
                print(f"  -> {spread_name}: needs {missing_in_basket}")


if __name__ == "__main__":
    compile_historical_state()