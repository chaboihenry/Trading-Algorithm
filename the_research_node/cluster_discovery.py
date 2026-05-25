import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

from the_utilities.paths import MODELS_DIR, CURATED_UNIVERSE_JSON, DISCOVERY_LEDGER_JSONL
from the_utilities.split_adjustment import apply_split_adjustment

# Discovery frequency: DAILY (changed from 5-min on 2026-05-24)
#
# The cointegration testing pipeline previously used 5-min bars. Empirical diagnosis showed
# this produced microstructure-driven false positives: Johansen trace statistics were 5-50x
# higher than daily on identical baskets (e.g., CVX/XLE/XOM showed trace=335 at 5-min vs
# 12.7 at daily against a critical value of 29.8), and OU R^2 was effectively zero at 5-min
# (e.g., GOOG/QQQ showed R^2=0.0003 at 5-min vs 0.95 at daily on the same pair).
#
# Switching to daily aligns with academic literature:
#   - Avellaneda & Lee (2010): "Statistical Arbitrage in the U.S. Equities Market" — daily returns
#   - Gatev, Goetzmann, Rouwenhorst (2006): "Pairs Trading: Performance of a Relative-Value
#     Arbitrage Rule" — daily prices
#   - Sarmento & Horta (2020): "Enhancing a Pairs Trading strategy with the application of
#     Machine Learning" — daily prices

# Local WSL2 ext4 storage
VAULT_ROOT = os.path.expanduser("~/quant_data/tick_data_storage")


def hurst_exponent(price_series, max_lag: int = 20):
    """Hurst exponent via variogram method. H < 0.5 = mean-reverting."""
    if len(price_series) < 100:
        return None
    lags = list(range(2, max_lag))
    tau = [np.sqrt(np.std(np.subtract(price_series[lag:], price_series[:-lag])))
           for lag in lags]
    # Drop non-positive tau so log is defined
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if len(valid) < 2:
        return None
    lags_valid, tau_valid = zip(*valid)
    poly = np.polyfit(np.log(lags_valid), np.log(tau_valid), 1)
    return float(poly[0] * 2.0)


def count_mean_crossings(spread):
    """Count zero-crossings of demeaned spread series."""
    demeaned = spread - spread.mean()
    sign_changes = np.diff(np.sign(demeaned.values if hasattr(demeaned, 'values') else demeaned))
    return int(np.sum(sign_changes != 0))


def _append_discovery_ledger(entry: dict):
    # One JSON line per cluster tested
    os.makedirs(os.path.dirname(DISCOVERY_LEDGER_JSONL), exist_ok=True)
    with open(DISCOVERY_LEDGER_JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")

def load_universe_list():
    # Ticker list from universe.txt
    try:
        with open('universe.txt', 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("[CRITICAL] universe.txt not found. Create it in the root directory.")
        return []

def load_daily_from_vault(tickers: list, lookback_days: int = 365):
    # Daily bars for PCA/DBSCAN clustering
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
    cutoff_str = cutoff_dt.strftime('%Y%m%d')

    daily_prices = {}

    for ticker in tickers:
        path = f"{VAULT_ROOT}/{ticker}/parquet/training_data"
        if not os.path.exists(path): continue

        daily_chunks = []
        for file in sorted(os.listdir(path)):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
            if file[:8] < cutoff_str:
                continue

            file_path = os.path.join(path, file)
            try:
                chunk = pd.read_parquet(file_path, columns=['timestamp', 'price'])
                if chunk.empty: continue
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
                daily = chunk.set_index('timestamp')['price'].resample('1D').last()
                daily = apply_split_adjustment(daily, ticker)
                daily_chunks.append(daily)
                del chunk
            except Exception:
                continue

        if not daily_chunks: continue
        series = pd.concat(daily_chunks).sort_index()
        daily_prices[ticker] = series[~series.index.duplicated(keep='last')].ffill()

    if not daily_prices:
        return pd.DataFrame()

    return pd.DataFrame(daily_prices).ffill().dropna()

def load_vault_data(cluster_tickers: list, lookback_days: int = 1260):
    # Daily bars for cointegration testing
    # 1260 trading days ≈ 5y. At 5-min freq, 90d gave ~7000 bars; at daily, 90d gives
    # only ~63 bars — below Johansen's reliable critical-value threshold.
    # 5y matches WRDS extent and Gatev (2006) / Sarmento & Horta (2020) / Avellaneda & Lee (2010, 252d windows).
    cutoff_dt = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=lookback_days)
    cutoff_str = cutoff_dt.strftime('%Y%m%d')

    ticker_series = {}

    for ticker in cluster_tickers:
        path = f"{VAULT_ROOT}/{ticker}/parquet/training_data"
        if not os.path.exists(path): continue

        resampled_chunks = []
        for file in sorted(os.listdir(path)):
            if not file.endswith('.parquet') or file.startswith('._'):
                continue
            if file[:8] < cutoff_str:
                continue

            file_path = os.path.join(path, file)
            try:
                chunk = pd.read_parquet(
                    file_path, columns=['timestamp', 'price'],
                    filters=[('timestamp', '>=', cutoff_dt)]
                )
                if chunk.empty: continue
                chunk['timestamp'] = pd.to_datetime(chunk['timestamp'], utc=True)
                bars = chunk.set_index('timestamp')['price'].resample('1D').last()
                bars = apply_split_adjustment(bars, ticker)
                resampled_chunks.append(bars)
                del chunk
            except Exception:
                continue

        if not resampled_chunks: continue
        series = pd.concat(resampled_chunks).sort_index()
        ticker_series[ticker] = series[~series.index.duplicated(keep='last')].ffill()

    if not ticker_series:
        return pd.DataFrame()

    # Align tickers on shared timestamps
    aligned = pd.DataFrame(ticker_series).dropna()
    return aligned

def test_cointegration(aligned_data: pd.DataFrame, tickers: list):
    # Johansen on in-sample + OOS spread ADF + OU regression R²
    empty = {
        'is_cointegrated': False,
        'half_life_days': None,
        'weights': None,
        'r_squared': None,
        'spread_adf_pvalue': None,
        'oos_spread_adf_pvalue': None,
        'hurst_exponent': None,
        'mean_crossings': None,
    }

    n = len(aligned_data)
    if n < 156:
        return empty

    # 70/30 in-sample vs held-out split for OOS persistence test
    split = int(n * 0.70)
    in_sample = aligned_data.iloc[:split]
    held_out = aligned_data.iloc[split:]

    if len(held_out) < 50:
        return empty

    res = coint_johansen(in_sample, det_order=0, k_ar_diff=1)
    trace_stat = res.lr1[0]
    crit_95 = res.cvt[0, 1]

    if trace_stat <= crit_95:
        return empty

    eigenvector = res.evec[:, 0]
    weights = dict(zip(tickers, eigenvector / eigenvector[0]))

    # In-sample spread stationarity (direct ADF)
    in_spread = in_sample.dot(eigenvector).dropna()
    spread_adf_p = float(adfuller(in_spread)[1])

    # Hurst and mean crossings on same in-sample spread
    hurst_val = hurst_exponent(in_spread.values)
    crossings = count_mean_crossings(in_spread)

    # OOS spread stationarity using same in-sample weights
    oos_spread = held_out.dot(eigenvector).dropna()
    oos_spread_adf_p = float(adfuller(oos_spread)[1])

    # OU half-life via OLS on spread differences (full sample)
    spread = aligned_data.dot(eigenvector)
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    ols = sm.OLS(spread_diff, sm.add_constant(spread_lag.loc[spread_diff.index])).fit()
    r_squared = float(ols.rsquared)

    lambda_val = ols.params.iloc[1]
    half_life = -np.log(2) / lambda_val if lambda_val < 0 else np.inf

    # At daily frequency, OU lambda is already per-day, so half_life is already in days
    return {
        'is_cointegrated': True,
        'half_life_days': half_life,
        'weights': weights,
        'r_squared': r_squared,
        'spread_adf_pvalue': spread_adf_p,
        'oos_spread_adf_pvalue': oos_spread_adf_p,
        'hurst_exponent': hurst_val,
        'mean_crossings': crossings,
    }

def enforce_websocket_limit(baskets: dict, max_tickers: int = 30):
    # Drop weakest (longest half-life) baskets until under ticker cap
    while True:
        unique_tickers = set()
        for data in baskets.values():
            unique_tickers.update(data['tickers'])

        if len(unique_tickers) <= max_tickers or not baskets:
            break

        weakest = max(baskets, key=lambda k: baskets[k].get('half_life', float('inf')))
        print(f"  >> [WEBSOCKET LIMIT] {len(unique_tickers)} tickers. "
              f"Dropping {weakest} (half-life: {baskets[weakest]['half_life']:.2f}d)...")
        baskets.pop(weakest)

    return baskets

def run_discovery_pipeline():
    universe = load_universe_list()
    if not universe: return

    print(f"\n=== ASYNC COMPUTE NODE: CLUSTER DISCOVERY ===")
    print(f"--- Processing Research Universe: {len(universe)} Assets ---")

    print("[SYSTEM] Loading daily bars from Parquet Vault for PCA clustering...")
    data = load_daily_from_vault(universe, lookback_days=365)

    if data.empty or len(data.columns) < 5:
        print("[CRITICAL] Insufficient vault data for clustering.")
        return

    print(f"[SYSTEM] Loaded {len(data)} daily bars across {len(data.columns)} assets.")

    # Cluster via PCA + DBSCAN on daily returns
    returns_t = data.pct_change().dropna().T
    scaled = StandardScaler().fit_transform(returns_t)
    pca = PCA(n_components=min(len(data.columns), 5)).fit_transform(scaled)
    clusters = DBSCAN(eps=1.2, min_samples=2).fit_predict(pca)

    results = pd.DataFrame({'Ticker': returns_t.index, 'Cluster': clusters})
    groups = results[results['Cluster'] != -1].groupby('Cluster')['Ticker'].apply(list)

    print(f"[SYSTEM] DBSCAN found {len(groups)} clusters. Testing cointegration...")

    confirmed_baskets = {}
    run_timestamp = pd.Timestamp.now(tz='UTC').isoformat()

    for _, cluster_tickers in groups.items():
        # Reject clusters > 12 tickers — Johansen critical values unreliable above n=12
        if len(cluster_tickers) > 12:
            _append_discovery_ledger({
                "timestamp": run_timestamp,
                "tickers": cluster_tickers,
                "status": "cluster_too_large_for_johansen",
                "cluster_size": len(cluster_tickers),
            })
            print(f"  >> [SIZE LIMIT] Skipping {len(cluster_tickers)}-ticker cluster (Johansen max=12)")
            continue

        aligned = load_vault_data(cluster_tickers)
        if aligned.empty:
            _append_discovery_ledger({
                "timestamp": run_timestamp,
                "tickers": cluster_tickers,
                "status": "no_vault_data",
            })
            continue

        coint_result = test_cointegration(aligned, cluster_tickers)
        is_coint = coint_result['is_cointegrated']
        hl_days = coint_result['half_life_days']
        weights = coint_result['weights']
        r_squared = coint_result['r_squared']
        spread_adf_p = coint_result['spread_adf_pvalue']
        oos_spread_adf_p = coint_result['oos_spread_adf_pvalue']
        hurst_val = coint_result['hurst_exponent']
        crossings = coint_result['mean_crossings']

        ledger_entry = {
            "timestamp": run_timestamp,
            "tickers": cluster_tickers,
            "is_cointegrated": bool(is_coint),
            "half_life_days": float(hl_days) if hl_days is not None else None,
            "weights": weights if weights else None,
            "r_squared": r_squared,
            "spread_adf_pvalue": spread_adf_p,
            "oos_spread_adf_pvalue": oos_spread_adf_p,
            "hurst_exponent": hurst_val,
            "mean_crossings": crossings,
        }

        if weights:
            abs_weights = [abs(w) for w in weights.values()]
            min_w = min(abs_weights)
            max_w = max(abs_weights)
            ledger_entry["min_max_weight_ratio"] = min_w / max_w if max_w > 0 else 0.0

            # Notional concentration — matches order router's 40% gate
            last_prices = aligned.iloc[-1]
            notional_per_leg = {t: abs(weights[t]) * last_prices[t] for t in weights}
            total_notional = sum(notional_per_leg.values())
            max_concentration = (max(notional_per_leg.values()) / total_notional
                                 if total_notional > 0 else 1.0)
            ledger_entry["max_notional_concentration"] = max_concentration
        else:
            ledger_entry["min_max_weight_ratio"] = None
            ledger_entry["max_notional_concentration"] = None

        # Filter chain: cointegration → ADF → OOS ADF → OU fit → half-life → weight ratio → notional concentration
        if not is_coint:
            ledger_entry["status"] = "not_cointegrated"
        elif spread_adf_p is None or spread_adf_p > 0.05:
            ledger_entry["status"] = f"spread_adf_failed (p={spread_adf_p})"
            print(f"  >> [SPREAD ADF] {'_'.join(cluster_tickers)}: "
                  f"in-sample p-value = {spread_adf_p}")
        elif oos_spread_adf_p is None or oos_spread_adf_p > 0.05:
            ledger_entry["status"] = f"oos_spread_adf_failed (p={oos_spread_adf_p})"
            print(f"  >> [OOS ADF] {'_'.join(cluster_tickers)}: "
                  f"held-out p-value = {oos_spread_adf_p}")
        elif r_squared is None or r_squared < 0.10:
            ledger_entry["status"] = f"ou_fit_too_weak (R2={r_squared})"
            print(f"  >> [OU FIT] {'_'.join(cluster_tickers)}: "
                  f"R² = {r_squared}")
        elif hurst_val is None or hurst_val >= 0.5:
            ledger_entry["status"] = f"hurst_not_mean_reverting (H={hurst_val})"
            print(f"  >> [HURST] {'_'.join(cluster_tickers)}: "
                  f"H = {hurst_val}")
        elif crossings is None or crossings < 12:
            ledger_entry["status"] = f"insufficient_mean_crossings ({crossings})"
            print(f"  >> [CROSSINGS] {'_'.join(cluster_tickers)}: "
                  f"in-sample crossings = {crossings}")
        # Half-life bounds: lower=1.0d (sub-bar half-lives meaningless at daily freq)
        # upper=30d (Sarmento & Horta 2020 cap; 15d was tuned for artificially short 5-min half-lives)
        elif not (1.0 <= hl_days <= 30.0):
            ledger_entry["status"] = f"half_life_out_of_range ({hl_days:.2f}d)"
        elif ledger_entry["min_max_weight_ratio"] is not None and ledger_entry["min_max_weight_ratio"] < 0.15:
            ledger_entry["status"] = "degenerate_hedge_ratio"
            print(f"  >> [DEGENERATE] {'_'.join(cluster_tickers)}: "
                  f"min/max weight ratio = {ledger_entry['min_max_weight_ratio']:.3f}")
        elif ledger_entry["max_notional_concentration"] is not None and ledger_entry["max_notional_concentration"] > 0.40:
            ledger_entry["status"] = f"notional_concentration_exceeded ({ledger_entry['max_notional_concentration']:.3f})"
            print(f"  >> [CONCENTRATION] {'_'.join(cluster_tickers)}: "
                  f"max notional weight = {ledger_entry['max_notional_concentration']:.1%}")
        else:
            spread_name = "_".join(cluster_tickers) + "_Spread"
            confirmed_baskets[spread_name] = {
                'tickers': cluster_tickers,
                'weights': weights,
                'half_life': hl_days
            }
            ledger_entry["status"] = "confirmed"
            print(f"  >> [CONFIRMED] {spread_name} | Half-life: {hl_days:.2f}d | "
                  f"weight ratio: {ledger_entry['min_max_weight_ratio']:.3f} | "
                  f"notional: {ledger_entry['max_notional_concentration']:.1%} | "
                  f"R²: {r_squared:.3f} | "
                  f"ADF p: {spread_adf_p:.4f}/{oos_spread_adf_p:.4f} | "
                  f"H: {hurst_val:.3f} | crossings: {crossings}")

        _append_discovery_ledger(ledger_entry)

    print(f"\n[SYSTEM] {len(confirmed_baskets)} cointegrated baskets confirmed.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Master ledger preserves historical baskets across runs
    ledger_path = os.path.join(MODELS_DIR, 'universe_baskets.json')
    ledger_payload = {"historical_basket_names": [], "baskets": {}}

    if os.path.exists(ledger_path):
        try:
            with open(ledger_path, 'r') as f:
                existing_data = json.load(f)
                if "historical_basket_names" in existing_data and "baskets" in existing_data:
                    ledger_payload = existing_data
                else:
                    ledger_payload["baskets"] = existing_data
        except Exception as e:
            print(f"[WARNING] Could not read existing ledger. Starting fresh. Error: {e}")

    current_time = pd.Timestamp.now(tz='UTC').isoformat()
    for basket_name, basket_data in confirmed_baskets.items():
        basket_data['last_seen'] = current_time
        ledger_payload["baskets"][basket_name] = basket_data

    ledger_payload["historical_basket_names"] = list(ledger_payload["baskets"].keys())

    temp_ledger_path = os.path.join(MODELS_DIR, 'universe_baskets_temp.json')
    with open(temp_ledger_path, 'w') as f:
        json.dump(ledger_payload, f, indent=4)
    os.replace(temp_ledger_path, ledger_path)

    print(f"[SYSTEM] Master Ledger updated. Tracking {len(ledger_payload['historical_basket_names'])} historical baskets.")

    confirmed_baskets = enforce_websocket_limit(confirmed_baskets, max_tickers=30)

    approved_tickers = set()
    for data in confirmed_baskets.values():
        approved_tickers.update(data['tickers'])

    if approved_tickers:
        payload = {
            "timestamp": current_time,
            "baskets": confirmed_baskets,
            "flat_list": list(approved_tickers)
        }

        temp_path = os.path.join(MODELS_DIR, 'curated_universe_temp.json')

        with open(temp_path, 'w') as f:
            json.dump(payload, f, indent=4)
        os.replace(temp_path, CURATED_UNIVERSE_JSON)

        print(f"[SUCCESS] Curated {len(approved_tickers)} tickers across {len(confirmed_baskets)} baskets for Live Execution.")
    else:
        print("[WARNING] No baskets survived. Execution node will hold cash.")

if __name__ == "__main__":
    run_discovery_pipeline()