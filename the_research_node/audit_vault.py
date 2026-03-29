import os
import pyarrow.dataset as ds

TICKER_TO_CHECK = 'V'
path = f"/Volumes/Vault/quant_data/tick data storage/{TICKER_TO_CHECK}/parquet"

if os.path.exists(path):
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(columns=['timestamp', 'price'])
    print(f"[SUCCESS] Found {len(table)} ticks for {TICKER_TO_CHECK} in the Valut.")
    print(f"Sample data :\n{table.to_pandas().head()}")
else:
    print(f"[ERROR] Path not found: {path}")
    print("Check if the SSD is named 'Vault' and the folder structure is correct.")


