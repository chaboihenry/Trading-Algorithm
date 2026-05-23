"""CRSP cfacpr-based split adjustment for raw TAQ trade prices.

Formula: adjusted_price = raw_price / cfacpr_at_that_date

The split_factors.parquet file is produced by the_research_node/fetch_split_factors.py
and contains columns: ticker, date, cfacpr from crsp.dsf.
"""

import os
import numpy as np
import pandas as pd

SPLIT_FACTORS_PATH = os.path.expanduser("~/quant_data/split_factors.parquet")

_SPLIT_FACTORS = None  # cached at module level


def _load_factors():
    # Load split factors parquet once on first access
    global _SPLIT_FACTORS
    if _SPLIT_FACTORS is None:
        if not os.path.exists(SPLIT_FACTORS_PATH):
            return None
        df = pd.read_parquet(SPLIT_FACTORS_PATH)
        df['date'] = pd.to_datetime(df['date'])
        _SPLIT_FACTORS = df
    return _SPLIT_FACTORS


def apply_split_adjustment(price_series, ticker):
    # Returns split-adjusted price series; if no CRSP data for ticker, returns unchanged
    factors = _load_factors()
    if factors is None:
        return price_series
    tf = factors[factors['ticker'] == ticker]
    if tf.empty:
        return price_series  # no CRSP coverage (e.g. VXX as an ETN)

    # Trading date for each timestamp in US Eastern time
    if price_series.index.tz is not None:
        ts_dates = price_series.index.tz_convert('America/New_York').normalize().tz_localize(None)
    else:
        ts_dates = price_series.index.normalize()

    # Per-date cfacpr lookup, forward-filled across non-trading days
    factor_lookup = tf.set_index('date')['cfacpr'].sort_index()
    unique_dates = pd.DatetimeIndex(np.unique(ts_dates))
    aligned = factor_lookup.reindex(unique_dates, method='ffill').fillna(1.0)

    # Divide raw by cfacpr
    return price_series / aligned.loc[ts_dates].values
