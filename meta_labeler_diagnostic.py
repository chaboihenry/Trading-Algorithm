import os
import json
import gc
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# PEPE: Import everything directly from the trainer so features are identical
from the_research_node.m1_xgboost_trainer import (
    construct_m1_dibs,
    get_offline_microstructure,
    get_daily_vol,
    find_optimal_d,
    apply_triple_barrier
)

UNIVERSE_PATH = "the_models/curated_universe.json"
MODELS_DIR = "the_models"


def load_model():
    """PEPE: Load the active model version from the ledger file"""
    version_path = os.path.join(MODELS_DIR, "active_model_version.txt")

    with open(version_path, "r") as f:
        lines = f.readlines()

    latest = lines[-1].strip()

    if "Model:" in latest:
        model_name = latest.split("Model:")[1].split("|")[0].strip()
    else:
        model_name = latest.strip()

    model_path = os.path.join(MODELS_DIR, model_name)
    print(f"[SYSTEM] Loading model: {model_path}")

    model = xgb.Booster()
    model.load_model(model_path)
    return model, model_name


def reconstruct_dataset():
    """
    PEPE: Carbon copy of the trainer's data pipeline (Steps 1-8).
    If this produces different features than training, the diagnosis is invalid.
    """
    with open(UNIVERSE_PATH, "r") as f:
        universe_data = json.load(f)

    baskets = universe_data.get("baskets", {})
    all_X, all_y, all_meta = [], [], []

    for name, data in baskets.items():
        tickers = data["tickers"]
        weights = data["weights"]
        anchor = tickers[0]

        print(f"\n[REBUILD] Processing: {name}")

        # 1. Build Structural DIBs — same function, same threshold
        dibs = construct_m1_dibs(anchor, threshold=50_000_000)
        if dibs.empty:
            print(f"  >> [SKIP] No DIBs for {anchor}")
            continue

        # 2. Load 1-min bars — safely filtering out massive duplicate files
        prices = {}
        for t in tickers:
            path = f"/Volumes/Vault/quant_data/tick data storage/{t}/parquet/training_data"
            if not os.path.exists(path):
                continue
                
            # Ignore the massive merged/wrds duplicates
            files = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if f.endswith(".parquet") and not f.startswith("._")
            ]
            
            if not files:
                continue
            df_t = pd.concat(
                [pd.read_parquet(f, columns=["timestamp", "price"]) for f in files]
            )
            df_t["timestamp"] = pd.to_datetime(df_t["timestamp"], utc=True)
            prices[t] = df_t.set_index("timestamp")["price"].resample("1min").last().ffill()

        if len(prices) != len(tickers):
            print(f"  >> [SKIP] Missing price data for {name}")
            continue

        # 3. Spread construction
        spread_continuous = sum(prices[t] * weights[t] for t in tickers).dropna()
        spread_continuous = spread_continuous[~spread_continuous.index.duplicated(keep="last")]

        unique_dib_index = dibs.index.drop_duplicates(keep="last")
        spread_dibs = spread_continuous.reindex(unique_dib_index, method="ffill").dropna()

        if len(spread_dibs) < 100:
            print(f"  >> [SKIP] Only {len(spread_dibs)} DIB samples for {name}")
            continue

        # 4. Features — identical to trainer
        vol = get_daily_vol(spread_dibs)
        opt_d, spread_fd = find_optimal_d(spread_dibs)
        print(f"  >> Stationarity at d={opt_d:.2f} | {len(spread_dibs)} DIB samples")

        z = (spread_dibs - spread_dibs.rolling(50).mean()) / spread_dibs.rolling(50).std()
        events = z[(z > 2.5) | (z < -2.5)].to_frame("z")
        events["trgt"] = vol.reindex(events.index).ffill()
        events = events.dropna()

        if events.empty:
            print(f"  >> [SKIP] No z-score events for {name}")
            continue

        # 5. Triple barrier labels
        labels = apply_triple_barrier(spread_dibs, events, pt_sl=[1, 2], t1=120)

        # 6. Microstructure
        df_anchor = prices[anchor].to_frame("price")
        df_anchor["size"] = 100
        micro = get_offline_microstructure(df_anchor)

        # 7. Feature matrix
        X = pd.DataFrame(index=labels.index)
        X["frac_diff"] = spread_fd.reindex(labels.index).ffill()
        X["volatility"] = vol.reindex(labels.index).ffill()
        X["signal_strength"] = events["z"]

        if not micro.empty:
            X["order_flow_imbalance"] = (
                micro["order_flow_imbalance"].reindex(labels.index).ffill().fillna(0)
            )
            X["effective_spread"] = (
                micro["effective_spread"].reindex(labels.index).ffill().fillna(0)
            )

        valid = X.dropna()
        y = labels["bin"].reindex(valid.index)

        # PEPE: Metadata for per-spread breakdown
        meta = pd.DataFrame(index=valid.index)
        meta["spread_name"] = name
        meta["z_score"] = events["z"].reindex(valid.index)
        meta["target_vol"] = events["trgt"].reindex(valid.index)

        all_X.append(valid)
        all_y.append(y)
        all_meta.append(meta)

        print(f"  >> [DONE] {name}: {len(valid)} setups | Win rate: {y.mean()*100:.1f}%")

        # PEPE: Free per-spread data before loading the next one
        del prices, spread_continuous, dibs, spread_dibs
        del vol, spread_fd, z, events, labels, micro, X, valid, y, meta
        gc.collect()

    if not all_X:
        print("[CRITICAL] No spreads produced valid features.")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    return pd.concat(all_X), pd.concat(all_y), pd.concat(all_meta)


def run_diagnostic():
    print("====== META-LABELER v2 PRECISION vs RECALL AUTOPSY ======\n")

    model, model_name = load_model()
    X, y, meta = reconstruct_dataset()

    if X.empty:
        print("[ABORT] No data to diagnose.")
        return

    print(f"\n[SYSTEM] Total setups reconstructed: {len(X)}")
    print(f"[SYSTEM] Class distribution: Wins={int(y.sum())} | Losses={int(len(y) - y.sum())}")
    print(f"[SYSTEM] Base win rate: {y.mean() * 100:.1f}%\n")

    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)

    # --- SECTION A: THRESHOLD SWEEP ---
    print("=" * 70)
    print("SECTION A: THRESHOLD SWEEP (Your backtest uses 0.55)")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1':<10} {'Trades':<10} {'Wins':<8}")
    print("-" * 70)

    best_f1 = 0
    best_thresh = 0.55

    for thresh in np.arange(0.40, 0.81, 0.05):
        preds = (probs >= thresh).astype(int)
        n_trades = preds.sum()

        if n_trades == 0:
            print(f"{thresh:<12.2f} {'N/A':<12} {'N/A':<10} {'N/A':<10} {0:<10}")
            continue

        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        wins = int((preds * y).sum())

        marker = " <-- CURRENT" if abs(thresh - 0.55) < 0.01 else ""
        print(f"{thresh:<12.2f} {prec:<12.3f} {rec:<10.3f} {f1:<10.3f} {n_trades:<10} {wins:<8}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\n[VERDICT] Optimal threshold by F1: {best_thresh:.2f} (F1={best_f1:.3f})")

    # --- SECTION B: CONFUSION MATRIX @ 0.55 ---
    print("\n" + "=" * 70)
    print("SECTION B: CONFUSION MATRIX @ 0.55 THRESHOLD")
    print("=" * 70)

    preds_current = (probs >= 0.55).astype(int)
    cm = confusion_matrix(y, preds_current)

    tn, fp, fn, tp = cm.ravel()
    print(f"\n  True Positives  (Good trades TAKEN):    {tp}")
    print(f"  False Positives (Bad trades TAKEN):     {fp}  <-- Precision problem")
    print(f"  False Negatives (Good trades MISSED):   {fn}  <-- Recall problem")
    print(f"  True Negatives  (Bad trades BLOCKED):   {tn}")

    # --- SECTION C: DIAGNOSIS ---
    print("\n" + "=" * 70)
    print("SECTION C: DIAGNOSIS")
    print("=" * 70)

    prec = precision_score(y, preds_current, zero_division=0)
    rec = recall_score(y, preds_current, zero_division=0)

    print(f"\n  Precision: {prec:.3f}  (Of trades taken, what % actually won)")
    print(f"  Recall:    {rec:.3f}  (Of all winning setups, what % did we catch)")

    try:
        auc = roc_auc_score(y, probs)
        print(f"  ROC-AUC:   {auc:.4f}")
    except ValueError:
        print("  ROC-AUC:   N/A (single class in dataset)")

    if prec > rec + 0.10:
        diagnosis = "LOW RECALL"
        explanation = (
            "The model is too conservative. It's blocking profitable trades.\n"
            "  FIX: Lower threshold, add sample weighting (Path 2) to amplify\n"
            "       high-magnitude wins the model currently ignores."
        )
    elif rec > prec + 0.10:
        diagnosis = "LOW PRECISION"
        explanation = (
            "The model is too aggressive. It's approving trades that lose.\n"
            "  FIX: Add macro regime features (Path 3) so the model learns\n"
            "       WHEN the market environment kills mean reversion."
        )
    else:
        diagnosis = "BALANCED (BOTH NEED WORK)"
        explanation = (
            "Neither precision nor recall dominates. The model lacks signal.\n"
            "  FIX: Apply both Path 2 (sample weighting) AND Path 3 (macro)\n"
            "       together. The feature space needs enrichment."
        )

    print(f"\n  >>> DIAGNOSIS: {diagnosis}")
    print(f"  >>> {explanation}")

    # --- SECTION D: PER-SPREAD BREAKDOWN ---
    print("\n" + "=" * 70)
    print("SECTION D: PER-SPREAD PERFORMANCE")
    print("=" * 70)

    meta["prob"] = probs
    meta["pred"] = preds_current
    meta["actual"] = y.values

    for spread_name in meta["spread_name"].unique():
        subset = meta[meta["spread_name"] == spread_name]
        n = len(subset)
        n_wins = int(subset["actual"].sum())
        n_taken = int(subset["pred"].sum())
        n_correct = int((subset["pred"] * subset["actual"]).sum())

        print(f"\n  {spread_name}:")
        print(f"    Total setups: {n} | Actual wins: {n_wins} | Trades taken: {n_taken} | Correct: {n_correct}")

        if n_taken > 0 and n_wins > 0:
            sp = precision_score(subset["actual"], subset["pred"], zero_division=0)
            sr = recall_score(subset["actual"], subset["pred"], zero_division=0)
            print(f"    Precision: {sp:.3f} | Recall: {sr:.3f}")
        else:
            print("    [INSUFFICIENT DATA]")

    print("\n====== DIAGNOSTIC COMPLETE ======")


if __name__ == "__main__":
    run_diagnostic()