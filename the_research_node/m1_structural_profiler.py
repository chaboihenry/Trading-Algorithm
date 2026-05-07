import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings

from the_research_node.m1_cluster_discovery import test_cointegration
from the_utilities.paths import MODELS_DIR, BACKTEST_PARQUET, STRUCTURAL_LIFECYCLE_JSON

warnings.filterwarnings('ignore')


def build_structural_history():
    print(f"[SYSTEM] Loading 5-year history from {BACKTEST_PARQUET}...")
    try:
        df_5m = pd.read_parquet(BACKTEST_PARQUET)
        if not isinstance(df_5m.index, pd.DatetimeIndex):
            df_5m.index = pd.to_datetime(df_5m.index)
        df_5m = df_5m.sort_index().ffill().dropna()
    except FileNotFoundError:
        print(f"[CRITICAL] Parquet file not found at {BACKTEST_PARQUET}")
        return

    # Downsample to Daily strictly for the PCA/DBSCAN economic clustering
    print("[SYSTEM] Resampling to daily bars for rapid PCA clustering...")
    df_1d = df_5m.resample('1D').last().dropna(how='all')

    # Walk-forward parameters (90-day lookback, sliding forward 30 days at a time)
    WINDOW_DAYS = 90
    STEP_DAYS = 30

    start_dates = pd.date_range(
        start=df_1d.index[0],
        end=df_1d.index[-1] - pd.Timedelta(days=WINDOW_DAYS),
        freq=f'{STEP_DAYS}D'
    )

    ledger = {}
    total_steps = len(start_dates)

    for idx, start_dt in enumerate(start_dates):
        end_dt = start_dt + pd.Timedelta(days=WINDOW_DAYS)
        print(f"\n--- Walk-Forward Window {idx+1}/{total_steps} [{start_dt.date()} to {end_dt.date()}] ---")

        # Slice the 90-day window
        slice_1d = df_1d.loc[start_dt:end_dt].dropna(axis=1)
        slice_5m = df_5m.loc[start_dt:end_dt]

        if slice_1d.empty or slice_5m.empty:
            continue

        # 1. Clustering Phase
        returns_t = slice_1d.pct_change().dropna().T
        scaled = StandardScaler().fit_transform(returns_t)
        pca = PCA(n_components=min(len(slice_1d.columns), 5)).fit_transform(scaled)
        clusters = DBSCAN(eps=1.2, min_samples=2).fit_predict(pca)

        results = pd.DataFrame({'Ticker': returns_t.index, 'Cluster': clusters})
        groups = results[results['Cluster'] != -1].groupby('Cluster')['Ticker'].apply(list)

        active_in_window = set()

        # 2. Cointegration Phase — uses the canonical Johansen test from m1_cluster_discovery
        for _, cluster_tickers in groups.items():
            aligned = slice_5m[cluster_tickers].dropna()
            if aligned.empty:
                continue

            is_coint, hl_days, weights = test_cointegration(aligned, cluster_tickers)

            if is_coint and (0.01 <= hl_days <= 15.0):
                spread_name = "_".join(cluster_tickers) + "_Spread"
                active_in_window.add(spread_name)

                if spread_name not in ledger:
                    ledger[spread_name] = {
                        "tickers": cluster_tickers,
                        "lifecycle": []
                    }

                lifecycle = ledger[spread_name]["lifecycle"]

                # New discovery, or the spread broke and has now re-cointegrated
                if not lifecycle or pd.to_datetime(lifecycle[-1]["end"]) < start_dt:
                    lifecycle.append({
                        "start": start_dt.isoformat(),
                        "end": end_dt.isoformat(),
                        "weights": weights,
                        "half_life": hl_days
                    })
                else:
                    # Spread survived — extend its life expectancy
                    lifecycle[-1]["end"] = end_dt.isoformat()
                    lifecycle[-1]["weights"] = weights
                    lifecycle[-1]["half_life"] = hl_days

        print(f"Validated {len(active_in_window)} structurally sound baskets.")

    # 3. Export the 5-Year Master Ledger
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(STRUCTURAL_LIFECYCLE_JSON, 'w') as f:
        json.dump(ledger, f, indent=4)

    print(f"\n[SUCCESS] Walk-Forward Complete. {len(ledger)} historical lifecycles saved to {STRUCTURAL_LIFECYCLE_JSON}.")


if __name__ == "__main__":
    build_structural_history()