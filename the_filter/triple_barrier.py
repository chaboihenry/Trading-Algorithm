import numpy as np
import pandas as pd
from numba import jit

from the_filter.volatility_estimator import get_daily_volatility
from the_filter.dib_constructor import construct_dibs

@jit(nopython=True)
def _numba_custom(price_diffs: np.ndarray, thresholds: np.ndarray):
    # c-compiled cusum filter
    t_events = []
    s_pos = 0.0
    s_neg = 0.0

    for i in range(len(price_diffs)):
        diff = price_diffs[i]
        thresh = thresholds[i]

        # calculate cumulative sum of upward and downward momentum
        s_pos = max(0.0, s_pos + diff)
        s_neg = min(0.0, s_neg + diff)

        # if momentum breaches dynamic volatility threshold, trigger an event
        if s_neg < -thresh:
            s_neg = 0.0
            t_events.append(i)
        elif s_pos > thresh: 
            s_pos = 0.0
            t_events.append(i)
    
    return t_events

def get_t_events(close: pd.Series, thresholds: pd.Series):
    # wrapper to align pandas indices with the numba numpy arrays
    price_diffs = close.diff().dropna()
    thresholds = thresholds.reindex(price_diffs.index).ffill().bfill()

    # get the integer indices of the events
    event_indices = _numba_custom(price_diffs.values, thresholds.values)

    # return the actual timestamps
    return price_diffs.index[event_indices]

def add_vertical_barrier(t_events: pd.DatetimeIndex, close: pd.Series, num_bars: int = 10):
    # map the vertical barrier (time-out) using integer indexing
    t_indices = close.index.searchsorted(t_events)
    v_barrier_indices = t_indices + num_bars

    # handle boundary conditions (if barrier projects past the end of dataset)
    v_barrier_indices = np.clip(v_barrier_indices, 0, len(close) - 1)

    return pd.Series(close.index[v_barrier_indices], index=t_events)

def get_events(close: pd.Series, t_events: pd.DatetimeIndex, pt_sl: list, target: pd.Series, min_ret: float, v_barriers: pd.Series):
    # core triple barrier logic
    # pt_sl is [profit_multiplier, stop_loss_multiplier]

    # filter out periods of extreme low volatility
    target = target.reindex(t_events).dropna()
    target = target[target > min_ret]

    events = pd.DataFrame(index=target.index)
    events['t1'] = v_barriers.reindex(target.index)
    events['trgt'] = target

    # matrix to hold the exact timestamps of barrier touches
    out = pd.DataFrame(index=events.index, columns=['pt', 'sl'])

    for loc, t1 in events['t1'].items():
        # slice the price path from the trigger to the vertical barrier
        path_prices = close.loc[loc:t1]
        path_returns = (path_prices / close.loc[loc]) -1

        # upper barrier (profit taking)
        if pt_sl[0] > 0: 
            upper_bound = pt_sl[0] * events.at[loc, 'trgt']
            pt_touches = path_returns[path_returns > upper_bound].index
            out.at[loc, 'pt'] = pt_touches[0] if len(pt_touches) > 0 else pd.NaT

        # lower barrier (stop-loss)
        if pt_sl[1] > 0:
            lower_bound = -pt_sl[1] * events.at[loc, 'trgt']
            sl_touches = path_returns[path_returns < lower_bound].index
            out.at[loc, 'sl'] = sl_touches[0] if len(sl_touches) > 0 else pd.NaT
    
    # force strictly datetime typing to prevent Timestamp vs Float comparison crashes
    out['pt'] = pd.to_datetime(out['pt'])
    out['sl'] = pd.to_datetime(out['sl'])
    out['t1'] = pd.to_datetime(events['t1'])
    
    # determine which barrier was hit first
    events['first_touch'] = out.min(axis=1)

    return events

def get_bins(events: pd.DataFrame, close: pd.Series):
    # meta-labeling. 1 for profit, -1 for loss, 0 for vertical barrier time-out.
    events_filtered = events.dropna(subset=['first_touch'])

    px_start = close.reindex(events_filtered.index)
    px_end = close.reindex(events_filtered['first_touch'].values)
    px_end.index = events_filtered.index

    ret = (px_end / px_start) - 1

    out = pd.DataFrame(index=events_filtered.index)
    out['ret'] = ret
    out['bin'] = np.sign(ret)

    # label vertical barrier time-outs as 0
    vertical_hits = events_filtered[events_filtered['first_touch'] == events_filtered['t1']].index
    out.loc[vertical_hits, 'bin'] = 0.0

    return out


if __name__ == "__main__":
    test_path = '/app/data/tick data storage/V/parquet/ticks.parquet'
    
    print("\n Reconstructing pristine DIBs...")
    dib_df = construct_dibs(test_path, threshold=50_000_000)
    
    print(" Calculating dynamic EWMA volatility...")
    vol_series = get_daily_volatility(dib_df['close'], span0=100)
    
    print(" Executing Numba CUSUM Filter (Finding structural breaks)...")
    t_events = get_t_events(dib_df['close'], vol_series)
    print(f"   -> Found {len(t_events)} actionable market events.")
    
    print(" Projecting Vertical Barriers (10-bar time limit)...")
    v_barriers = add_vertical_barrier(t_events, dib_df['close'], num_bars=10)
    
    print(" Executing Triple-Barrier Search (PT: 2x Vol, SL: 1x Vol)...")
    # we use an asymmetric payoff: taking profit at 2x volatility, cutting losses at 1x volatility
    events = get_events(dib_df['close'], t_events, pt_sl=[2.0, 1.0], target=vol_series, min_ret=0.001, v_barriers=v_barriers)
    
    print(" Generating Meta-Labels...")
    labels = get_bins(events, dib_df['close'])
    
    print("\n=== FINAL META-LABELS (FEATURES FOR XGBOOST) ===")
    print(labels.head(10))
    print("\n=== LABEL DISTRIBUTION ===")
    print(labels['bin'].value_counts())

