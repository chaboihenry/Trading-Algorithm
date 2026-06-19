"""
Rolling Johansen cointegration on the fixed-income basket that passed all
filters except half-life. Question: is the OOS-robust cointegration STABLE
across time, or a single-window artifact?

Walks a 252-day window across the full daily history, runs Johansen on each,
records whether the basket is cointegrated (trace stat > 95% critical value)
and the half-life in that window.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import statsmodels.api as sm

from the_research_node.cluster_discovery import load_vault_data

BASKET = ["AGG", "BND", "EMB", "HYG", "HYLB", "IEF",
          "JNK", "LQD", "SHY", "TIP", "TLT"]
WINDOW = 252   # 1 trading year
STEP = 21      # slide ~monthly


def basket_half_life(spread: pd.Series) -> float:
    # OU half-life in days from daily spread (lambda already per-day)
    lag = spread.shift(1).dropna()
    diff = spread.diff().dropna()
    common = diff.index.intersection(lag.index)
    if len(common) < 30:
        return np.nan
    ols = sm.OLS(diff.loc[common], sm.add_constant(lag.loc[common])).fit()
    lam = ols.params.iloc[1]
    return float(-np.log(2) / lam) if lam < 0 else np.nan


def main():
    prices = load_vault_data(BASKET, lookback_days=1825).dropna()
    print(f"Loaded {prices.shape[0]} daily bars, "
          f"{prices.index.min().date()} to {prices.index.max().date()}")
    print(f"Window={WINDOW}d, step={STEP}d, basket={len(BASKET)} assets\n")

    results = []
    for end in range(WINDOW, len(prices) + 1, STEP):
        win = prices.iloc[end - WINDOW:end]
        try:
            jres = coint_johansen(win, det_order=0, k_ar_diff=1)
            trace = jres.lr1[0]
            crit95 = jres.cvt[0, 1]
            coint = trace > crit95
            eig = jres.evec[:, 0]
            spread = win.dot(eig)
            hl = basket_half_life(spread)
        except Exception as e:
            trace, crit95, coint, hl = np.nan, np.nan, False, np.nan
        results.append({
            "window_end": win.index[-1].date(),
            "trace": trace, "crit95": crit95,
            "cointegrated": coint, "half_life_d": hl,
        })

    df = pd.DataFrame(results)
    n = len(df)
    pct_coint = 100 * df["cointegrated"].mean()
    hl_valid = df.loc[df["cointegrated"], "half_life_d"].dropna()

    print(f"Windows tested: {n}")
    print(f"% cointegrated (trace > 95% crit): {pct_coint:.1f}%")
    if len(hl_valid):
        print(f"Half-life (days) when cointegrated: "
              f"min={hl_valid.min():.2f}, median={hl_valid.median():.2f}, "
              f"max={hl_valid.max():.2f}")
        print(f"  windows with half-life >= 1.0d: "
              f"{100*(hl_valid >= 1.0).mean():.1f}%")
    print()
    # Show the trajectory
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
