"""
Rolling cointegration diagnostic.

For each candidate basket, slides a fixed-length window across the full price
history and runs Engle-Granger cointegration on each window. Tells you whether
a basket is:
  - Stably cointegrated across most windows  -> real signal
  - Cointegrated in some windows but not others -> regime-dependent
  - Never cointegrated -> small-sample noise from cluster_discovery

This is the diagnostic that distinguishes "the strategy works but not over the
5-year window we tested" from "the strategy doesn't work in this regime at all."

Reference:
  Avellaneda & Lee (2010): 252-day rolling formation windows
  Krauss (2017): documented post-2010 decay in pairs cointegration
  Vidyamurthy (2004): correlation != cointegration
"""
import os
import sys
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
import matplotlib
matplotlib.use("Agg")  # Headless: save to PNG, no display
import matplotlib.pyplot as plt

from the_research_node.cluster_discovery import load_vault_data
from the_utilities.paths import LOGS_DIR


# Window size: 1 trading year per Avellaneda & Lee (2010)
WINDOW_DAYS = 252
# Step size: slide by ~1 month (smaller = more windows, more computation)
STEP_DAYS = 21
# Minimum dropna observations before Engle-Granger is reliable
MIN_OBS = 200

# Candidate baskets to diagnose
BASKETS = {
    "HYG_TLT":   ["HYG", "TLT"],   # 252-day near-passer (OOS p=0.33)
    "BAC_JPM":   ["BAC", "JPM"],   # Control: confirmed non-cointegrated
    "HD_LOW":    ["HD", "LOW"],    # Textbook pair (home improvement)
    "MA_V":      ["MA", "V"],      # Textbook pair (payment networks)
}


def rolling_engle_granger(prices: pd.DataFrame,
                          window: int = WINDOW_DAYS,
                          step: int = STEP_DAYS) -> pd.DataFrame:
    """
    Rolling Engle-Granger test for a 2-asset pair.
    Returns DataFrame with (window_start, window_end, p_value, n_obs).
    """
    if prices.shape[1] != 2:
        raise ValueError("Engle-Granger requires exactly 2 assets")

    # Drop weekends/holidays (NaN from calendar-daily resample)
    prices = prices.dropna()
    a, b = prices.columns
    results = []

    # Slide window across the full history
    for end_idx in range(window, len(prices) + 1, step):
        win = prices.iloc[end_idx - window: end_idx]
        if len(win) < MIN_OBS:
            continue
        try:
            _, pvalue, _ = coint(win[a], win[b])
        except Exception:
            pvalue = np.nan
        results.append({
            "window_start": win.index[0],
            "window_end":   win.index[-1],
            "p_value":      float(pvalue),
            "n_obs":        len(win),
        })

    return pd.DataFrame(results)


def plot_rolling_pvalue(results: pd.DataFrame,
                        basket_name: str,
                        output_dir: str) -> str:
    """Save rolling p-value plot, return path."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results["window_end"], results["p_value"], linewidth=1.0)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="p=0.05")
    ax.axhline(0.10, color="orange", linestyle=":", linewidth=1, label="p=0.10")
    ax.set_title(f"Rolling Engle-Granger Cointegration: {basket_name}")
    ax.set_xlabel("Window End Date")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, f"rolling_coint_{basket_name}.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def summarize(results: pd.DataFrame) -> dict:
    """Compact summary statistics over all windows."""
    valid = results.dropna(subset=["p_value"])
    if len(valid) == 0:
        return {"windows": 0}
    return {
        "windows":       len(valid),
        "pct_below_05":  100 * (valid["p_value"] < 0.05).mean(),
        "pct_below_10":  100 * (valid["p_value"] < 0.10).mean(),
        "p_min":         valid["p_value"].min(),
        "p_median":      valid["p_value"].median(),
        "p_mean":        valid["p_value"].mean(),
        "p_max":         valid["p_value"].max(),
    }


def run_diagnostic():
    print("=== ROLLING COINTEGRATION DIAGNOSTIC ===")
    print(f"Window: {WINDOW_DAYS} trading days, Step: {STEP_DAYS} days")
    print(f"Threshold: p < 0.05 is conventional cointegration significance\n")

    os.makedirs(LOGS_DIR, exist_ok=True)

    for basket_name, tickers in BASKETS.items():
        print(f"--- {basket_name} ({'/'.join(tickers)}) ---")
        try:
            # Load full available history (1825 calendar days ~= 5 trading years)
            prices = load_vault_data(tickers, lookback_days=1825)
        except Exception as e:
            print(f"[ERROR] Could not load data: {e}\n")
            continue

        print(f"Data: {prices.shape[0]} bars, "
              f"{prices.index.min().date()} to {prices.index.max().date()}")

        if len(tickers) != 2:
            print(f"[SKIP] Multi-asset baskets not supported yet (need Johansen)\n")
            continue

        results = rolling_engle_granger(prices)
        s = summarize(results)
        if s["windows"] == 0:
            print(f"[WARNING] No valid windows produced\n")
            continue

        print(f"Windows tested:      {s['windows']}")
        print(f"% p<0.05:            {s['pct_below_05']:.1f}%")
        print(f"% p<0.10:            {s['pct_below_10']:.1f}%")
        print(f"p-value min/med/max: "
              f"{s['p_min']:.3f} / {s['p_median']:.3f} / {s['p_max']:.3f}")

        plot_path = plot_rolling_pvalue(results, basket_name, LOGS_DIR)
        csv_path = os.path.join(LOGS_DIR, f"rolling_coint_{basket_name}.csv")
        results.to_csv(csv_path, index=False)
        print(f"Plot: {plot_path}")
        print(f"CSV:  {csv_path}\n")


if __name__ == "__main__":
    run_diagnostic()
