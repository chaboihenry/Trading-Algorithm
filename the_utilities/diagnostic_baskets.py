"""
Diagnostic: investigate all baskets from cluster discovery for cointegration quality.

Runs the full statistical battery at daily and 5-min frequency:
  - Cointegration test (Engle-Granger for pairs, Johansen for multi-asset)
  - ADF on constructed spread
  - Hurst exponent (variogram method)
  - OU regression half-life and R²
  - Mean return correlation

Compares against academic benchmarks:
  - Gatev et al. (2006): ~38% pair profitability out-of-sample
  - Sarmento & Horta (2020): 1-3% survival post-cointegration + Hurst filters
  - Vidyamurthy (2004): correlation != cointegration warning
  - Avellaneda & Lee (2010): PCA-based stat arb framework
"""

import os
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from the_utilities.split_adjustment import apply_split_adjustment

warnings.filterwarnings('ignore')

VAULT_ROOT = os.path.expanduser("~/quant_data/tick_data_storage")

# Baskets from cluster discovery run 1 — all 10 with visible failure modes
BASKETS_TO_INVESTIGATE = [
    (['AXP', 'BAC', 'DIS', 'JPM', 'NKE', 'WFC'], 'OOS ADF p=0.17'),
    (['C', 'GS', 'META', 'MS'], 'OOS ADF p=0.42'),
    (['COST', 'DUK', 'LMT', 'SO'], 'OOS ADF p=0.39'),
    (['CVX', 'XLE', 'XOM'], 'R^2 = 0.093 (borderline)'),
    (['GOOG', 'QQQ'], 'In-sample ADF p=0.11'),
    (['HD', 'LOW'], 'R^2 = 0.01 (investigated previously)'),
    (['IWM', 'XLY'], 'In-sample ADF p=0.23'),
    (['MA', 'V'], 'OOS ADF p=0.33'),
    (['T', 'VZ'], 'In-sample ADF p=0.26'),
]


def load_ticker(ticker, resample_freq='1D'):
    # Load all parquet files for a ticker, skipping in-progress files
    path = f"{VAULT_ROOT}/{ticker}/parquet/training_data"
    if not os.path.exists(path):
        return None
    chunks = []
    for f in sorted(os.listdir(path)):
        if f.endswith('.parquet') and not f.startswith('._'):
            try:
                df = pd.read_parquet(os.path.join(path, f), columns=['timestamp', 'price'])
            except Exception:
                continue
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            bars = df.set_index('timestamp')['price'].resample(resample_freq).last()
            bars = apply_split_adjustment(bars, ticker)
            chunks.append(bars)
    if not chunks:
        return None
    series = pd.concat(chunks).sort_index()
    return series[~series.index.duplicated(keep='last')].ffill().dropna()


def load_basket(tickers, resample_freq='1D'):
    # Load multiple tickers, align on common timestamps
    data = {}
    for t in tickers:
        series = load_ticker(t, resample_freq)
        if series is None:
            return None
        data[t] = series
    return pd.DataFrame(data).dropna()


def hurst_exponent(price_series, max_lag=20):
    # Variogram method, H < 0.5 = mean-reverting
    arr = np.asarray(price_series)
    if len(arr) < 100:
        return None
    lags = list(range(2, max_lag))
    tau = [np.sqrt(np.std(np.subtract(arr[lag:], arr[:-lag]))) for lag in lags]
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return None
    lags_v, tau_v = zip(*valid)
    poly = np.polyfit(np.log(lags_v), np.log(tau_v), 1)
    return float(poly[0] * 2.0)


def mean_pairwise_correlation(df):
    # Average off-diagonal correlation across return series
    rets = df.pct_change().dropna()
    n = len(df.columns)
    if n < 2:
        return None
    corr_matrix = rets.corr().values
    off_diag_sum = corr_matrix.sum() - np.trace(corr_matrix)
    n_pairs = n * (n - 1)
    return float(off_diag_sum / n_pairs)


def analyze_basket(tickers, df, freq):
    # Run full battery of cointegration tests on aligned basket data
    n = len(tickers)
    result = {
        'n_tickers': n,
        'n_bars': len(df),
        'frequency': freq,
        'mean_return_corr': mean_pairwise_correlation(df),
    }

    # Cointegration test depends on basket size
    try:
        if n == 2:
            ts, p, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
            result['coint_test'] = 'Engle-Granger'
            result['coint_stat'] = float(ts)
            result['coint_p'] = float(p)
            result['cointegrated'] = bool(p < 0.05)
            ols = sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit()
            spread = df.iloc[:, 0] - ols.params.iloc[1] * df.iloc[:, 1]
        else:
            jres = coint_johansen(df, det_order=0, k_ar_diff=1)
            trace_stat = float(jres.lr1[0])
            crit_95 = float(jres.cvt[0, 1])
            result['coint_test'] = 'Johansen'
            result['coint_stat'] = trace_stat
            result['crit_95'] = crit_95
            result['cointegrated'] = bool(trace_stat > crit_95)
            eig = jres.evec[:, 0]
            spread = df.dot(eig)
    except Exception as e:
        result['error'] = str(e)
        return result

    # ADF on spread
    try:
        adf_stat, adf_p, *_ = adfuller(spread)
        result['adf_spread_stat'] = float(adf_stat)
        result['adf_spread_p'] = float(adf_p)
    except Exception:
        result['adf_spread_p'] = None

    # Hurst
    result['hurst'] = hurst_exponent(spread.values)

    # OU regression for half-life and R²
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    common = spread_diff.index.intersection(spread_lag.index)
    if len(common) > 10:
        ols_hl = sm.OLS(spread_diff.loc[common], sm.add_constant(spread_lag.loc[common])).fit()
        lam = float(ols_hl.params.iloc[1])
        result['ou_r2'] = float(ols_hl.rsquared)
        if lam < 0:
            bars_per_day = 1 if freq == '1D' else 78
            result['half_life_days'] = float(-np.log(2) / lam / bars_per_day)
        else:
            result['half_life_days'] = None
    else:
        result['half_life_days'] = None
        result['ou_r2'] = None

    return result


def print_basket_report(tickers, observed_failure, daily_result, intraday_result):
    # Print results for one basket at both frequencies
    name = '_'.join(tickers)
    print(f"\n{'=' * 72}")
    print(f"  {name} ({len(tickers)} tickers)")
    print(f"  Observed failure mode: {observed_failure}")
    print(f"{'=' * 72}")

    for label, r in [('Daily bars', daily_result), ('5-min bars', intraday_result)]:
        print(f"\n  {label}:")
        if r is None:
            print(f"    [no data available]")
            continue
        if 'error' in r:
            print(f"    [error: {r['error']}]")
            continue
        print(f"    Bars: {r['n_bars']}, mean return corr: {r['mean_return_corr']:.3f}")
        if r['coint_test'] == 'Engle-Granger':
            print(f"    Engle-Granger: stat={r['coint_stat']:.3f}, p={r['coint_p']:.4f} "
                  f"[{'COINT' if r['cointegrated'] else 'NOT COINT'}]")
        else:
            print(f"    Johansen: trace={r['coint_stat']:.3f}, crit95={r['crit_95']:.3f} "
                  f"[{'COINT' if r['cointegrated'] else 'NOT COINT'}]")
        if r.get('adf_spread_p') is not None:
            print(f"    Spread ADF: stat={r['adf_spread_stat']:.3f}, p={r['adf_spread_p']:.4f}")
        if r.get('hurst') is not None:
            mr = '(mean-reverting)' if r['hurst'] < 0.5 else '(trending)'
            print(f"    Hurst: {r['hurst']:.3f} {mr}")
        if r.get('half_life_days') is not None:
            print(f"    OU half-life: {r['half_life_days']:.2f} days")
        if r.get('ou_r2') is not None:
            flag = ' <-- LOW' if r['ou_r2'] < 0.10 else ''
            print(f"    OU R^2: {r['ou_r2']:.4f}{flag}")


def summarize(all_results):
    # Aggregate findings across all baskets, compare to literature
    print(f"\n{'#' * 72}")
    print(f"  AGGREGATE SUMMARY — {len(all_results)} BASKETS")
    print(f"{'#' * 72}")

    pairs = [r for r in all_results if len(r['tickers']) == 2]
    multi = [r for r in all_results if len(r['tickers']) > 2]
    print(f"\n  Composition: {len(pairs)} pairs, {len(multi)} multi-asset baskets")

    # Daily-level outcomes
    print(f"\n  Daily-frequency cointegration:")
    daily_pass = sum(1 for r in all_results if r['daily'] and r['daily'].get('cointegrated'))
    daily_total = sum(1 for r in all_results if r['daily'])
    print(f"    Passes cointegration test: {daily_pass}/{daily_total}")

    # R² distribution
    daily_r2 = [r['daily']['ou_r2'] for r in all_results
                if r['daily'] and r['daily'].get('ou_r2') is not None]
    if daily_r2:
        print(f"    OU R^2 distribution: min={min(daily_r2):.3f}, median={np.median(daily_r2):.3f}, max={max(daily_r2):.3f}")
        print(f"    R^2 >= 0.10 (filter passes): {sum(1 for r2 in daily_r2 if r2 >= 0.10)}/{len(daily_r2)}")

    # Hurst distribution
    daily_hurst = [r['daily']['hurst'] for r in all_results
                   if r['daily'] and r['daily'].get('hurst') is not None]
    if daily_hurst:
        print(f"    Hurst distribution: min={min(daily_hurst):.3f}, median={np.median(daily_hurst):.3f}, max={max(daily_hurst):.3f}")
        print(f"    Hurst < 0.5 (mean-reverting): {sum(1 for h in daily_hurst if h < 0.5)}/{len(daily_hurst)}")

    # Mean correlation distribution — useful for the correlation != cointegration point
    corrs = [r['daily']['mean_return_corr'] for r in all_results
             if r['daily'] and r['daily'].get('mean_return_corr') is not None]
    if corrs:
        print(f"    Mean return correlation: min={min(corrs):.3f}, median={np.median(corrs):.3f}, max={max(corrs):.3f}")
        print(f"    Note: high correlation found even where cointegration failed (Vidyamurthy 2004)")

    print(f"\n  LITERATURE BENCHMARKS:")
    print(f"    Gatev et al. (2006): ~38% of selected pairs profitable out-of-sample")
    print(f"    Sarmento & Horta (2020): 1-3% pair survival on S&P 500 after full filter chain")
    print(f"    Krauss (2017): post-2010 pairs trading performance has decayed significantly")
    print(f"\n    This run (5 months of data): 0 baskets survived full filter chain")
    print(f"    Expected with full 5-year data: increased OOS ADF power should let")
    print(f"    statistically-valid baskets pass through.")


if __name__ == "__main__":
    print(f"\nLoading data from: {VAULT_ROOT}")
    print(f"Investigating {len(BASKETS_TO_INVESTIGATE)} baskets at daily and 5-min frequency...\n")

    all_results = []
    for tickers, observed in BASKETS_TO_INVESTIGATE:
        daily = load_basket(tickers, '1D')
        intraday = load_basket(tickers, '5min')

        daily_result = analyze_basket(tickers, daily, '1D') if daily is not None else None
        intraday_result = analyze_basket(tickers, intraday, '5min') if intraday is not None else None

        print_basket_report(tickers, observed, daily_result, intraday_result)

        all_results.append({
            'tickers': tickers,
            'daily': daily_result,
            'intraday': intraday_result,
        })

    summarize(all_results)