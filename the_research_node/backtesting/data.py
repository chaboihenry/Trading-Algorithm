"""
Stat-arb backtester — LAYER 1: data loading + spread reconstruction.

This layer only rebuilds the cointegrated spread from raw bars so later
layers (signals, position simulation, costs, metrics) have a clean,
fully-aligned price panel to work from. No trading logic lives here.

The basket and weights are the Johansen cointegrating vector emitted by
cluster_discovery; AGG is normalized to 1.0 so the spread is expressed in
AGG-equivalent units.
"""

import os
import pandas as pd

from the_utilities.split_adjustment import apply_split_adjustment
from the_utilities.paths import VAULT_ROOT

BASKET = ["AGG", "BND", "EMB", "HYG", "HYLB", "IEF", "JNK", "LQD", "SHY", "TIP", "TLT"]

WEIGHTS = {  # Johansen cointegrating vector from cluster_discovery
    "AGG": 1.0,
    "BND": -6.0787437372573,
    "EMB": -0.5439938231177188,
    "HYG": 27.50283717187229,
    "HYLB": -11.025393795466792,
    "IEF": 2.3856873363868076,
    "JNK": -14.495799077802396,
    "LQD": -0.01453598889961951,
    "SHY": 0.3758055018739757,
    "TIP": -0.36786943578717146,
    "TLT": 0.4963985651419994,
}


def load_basket_bars(tickers: list, freq: str = "1h") -> pd.DataFrame:
    # Build one fully-aligned price panel: a leg missing any bar would
    # silently distort the spread, so we inner-join on common timestamps.
    data = {}
    for ticker in tickers:
        path = f"{VAULT_ROOT}/{ticker}/parquet/training_data"
        if not os.path.exists(path):
            continue
        chunks = []
        for f in sorted(os.listdir(path)):
            # Skip macOS resource-fork sidecars left on the SSD ("._" prefix)
            if f.endswith(".parquet") and not f.startswith("._"):
                try:
                    df = pd.read_parquet(os.path.join(path, f), columns=["timestamp", "price"])
                except Exception:
                    continue
                # Tick vault is huge; reparsing an already tz-aware datetime
                # column iterates billions of values for nothing — skip it.
                ts = df["timestamp"]
                if not pd.api.types.is_datetime64_any_dtype(ts):
                    ts = pd.to_datetime(ts, utc=True)
                df = df.assign(timestamp=ts)
                # .last() = last trade in the bar, the tradable close for that interval
                bars = df.set_index("timestamp")["price"].resample(freq).last()
                bars = apply_split_adjustment(bars, ticker)
                chunks.append(bars)
        if not chunks:
            continue
        # Overlapping month files can repeat the boundary bar; keep the latest
        series = pd.concat(chunks).sort_index()
        series = series[~series.index.duplicated(keep="last")].ffill()
        data[ticker] = series

    # dropna() across columns enforces that every surviving row has all legs
    return pd.DataFrame(data).dropna()


def load_basket_bars_cached(tickers: list, freq: str = "1h", rebuild: bool = False) -> pd.DataFrame:
    # The tick->hourly resample is the expensive step; persist its result so
    # only the first run pays for it and later runs are near-instant reads.
    cache_path = os.path.join(
        os.path.dirname(VAULT_ROOT), "backtest_cache", f"basket_hourly_{freq}.parquet"
    )
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if not rebuild and os.path.exists(cache_path):
        print(f"[CACHE] loaded {cache_path}")
        return pd.read_parquet(cache_path)

    prices = load_basket_bars(tickers, freq)
    prices.to_parquet(cache_path, compression="zstd")
    print(f"[CACHE] rebuilt {cache_path}")
    return prices


def build_spread(prices: pd.DataFrame, weights: dict) -> pd.Series:
    # Weighted sum of legs = the cointegrating combination that mean-reverts
    spread = sum(prices[ticker] * weights[ticker] for ticker in weights)
    return spread.rename("spread")


if __name__ == "__main__":
    # Data-layer smoke test only — nothing is traded here.
    cache_path = os.path.join(
        os.path.dirname(VAULT_ROOT), "backtest_cache", "basket_hourly_1h.parquet"
    )
    print(f"Cache path: {cache_path}")
    # First run builds the cache; subsequent runs read it and stay fast.
    prices = load_basket_bars_cached(BASKET, "1h")
    print(f"Loaded panel shape: {prices.shape}")
    if not prices.empty:
        print(f"Date range: {prices.index.min()} -> {prices.index.max()}")

    spread = build_spread(prices, WEIGHTS)
    print("\nSpread describe():")
    print(spread.describe())

    # Mean crossings = sign changes of the demeaned spread; a rough sanity
    # check that the combination actually oscillates rather than drifts.
    centered = spread - spread.mean()
    crossings = int((centered.shift(1) * centered < 0).sum())
    print(f"\nMean crossings: {crossings}")
