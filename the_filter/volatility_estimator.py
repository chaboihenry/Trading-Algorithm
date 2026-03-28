import pandas as pd
import numpy as np

def get_daily_volatility(close_series: pd.Series, span0: int = 100):
    # calculate dynamic volatility to size profit/stop-loss barriers
    # using exponentially weighted moving standard deviation

    # calculate tick-to-tick returns
    returns = close_series.pct_change()

    # calculate EWM std, span0 is lookback window for the decay factor
    ewma_vol = returns.ewm(span=span0).std()

    # shift forward by 1 to avoid looking into the future
    # the volatility used to size today's barrier must be based on yesterday's data
    ewma_vol = ewma_vol.shift(1)

    return ewma_vol.dropna()


if __name__ == "__main__":
    from the_filter.dib_constructor import construct_dibs
    test_path = '/app/data/tick data storage/V/parquet/ticks.parquet'
    print("\n Reconstructing pristine DIBs...")
    dib_df = construct_dibs(test_path, threshold=50_000_000)
    print("\n Calculating dynamic EWMA volatility...")
    vol_series = get_daily_volatility(dib_df['close'], span0=100)
    print("\n===== Dynamic Barrier Widths (Volatility) ====")
    print(vol_series.tail(10))


