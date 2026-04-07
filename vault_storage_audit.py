# the_utilities/vault_storage_audit.py
# PEPE: Maps out exactly what's in each ticker's parquet folder
# Run this to understand the storage layout before touching the diagnostic

import os
import pandas as pd
import pyarrow.parquet as pq

VAULT_BASE = "/Volumes/Vault/quant_data/tick data storage"

# PEPE: Only audit a handful of tickers to keep it fast
SAMPLE_TICKERS = ["AVGO", "BAC", "AAPL", "SPY", "GOOG"]


def audit_ticker(ticker):
    path = os.path.join(VAULT_BASE, ticker, "parquet")
    if not os.path.exists(path):
        print(f"\n[MISSING] {ticker}: No parquet folder found")
        return

    print(f"\n{'=' * 70}")
    print(f"TICKER: {ticker}")
    print(f"{'=' * 70}")

    files = sorted(os.listdir(path))
    total_size = 0
    total_rows = 0

    print(f"{'Filename':<50} {'Size (MB)':<12} {'Rows':<12} {'Columns'}")
    print("-" * 100)

    for f in files:
        file_path = os.path.join(path, f)

        # PEPE: Skip non-files (directories, symlinks)
        if not os.path.isfile(file_path):
            continue

        # PEPE: Skip macOS ghost files
        if f.startswith("._"):
            continue

        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        total_size += size_mb

        # PEPE: Read only metadata to avoid loading data into RAM
        try:
            meta = pq.read_metadata(file_path)
            rows = meta.num_rows
            total_rows += rows

            # Grab column names from the schema
            schema = pq.read_schema(file_path)
            cols = schema.names
        except Exception as e:
            rows = "ERROR"
            cols = str(e)

        print(f"{f:<50} {size_mb:<12.2f} {str(rows):<12} {cols}")

    print("-" * 100)
    print(f"{'TOTAL':<50} {total_size:<12.2f} {total_rows:<12}")
    print(f"File count: {len([f for f in files if not f.startswith('._')])}")


if __name__ == "__main__":
    print("====== VAULT PARQUET STORAGE AUDIT ======")

    for ticker in SAMPLE_TICKERS:
        audit_ticker(ticker)

    print("\n====== AUDIT COMPLETE ======")