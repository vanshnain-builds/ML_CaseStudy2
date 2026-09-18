"""
Credit Card Fraud Detection — XGBoost + SMOTE Case Study
==========================================================
Full reproducible pipeline for the IEEE-CIS Fraud Detection dataset (Kaggle).

Expects, in a `data/` subfolder next to this script:
    data/train_transaction.csv
    data/train_identity.csv
(the standard files from the Kaggle "IEEE-CIS Fraud Detection" competition)

Produces:
    data/features_clean.parquet      - cleaned, fully-numeric feature matrix
    data/model_baseline.json         - XGBoost booster, class-weighted
    data/model_smote.json            - XGBoost booster, undersample+SMOTE trained
    data/results_baseline.json       - ROC-AUC / PR-AUC for the baseline model
    data/results_smote.json          - ROC-AUC / PR-AUC for the SMOTE model
    data/threshold_summary.json      - precision/recall/F1 at 3 candidate thresholds
    data/top_features_gain.json      - top-20 features by XGBoost gain importance
    data/top_features_shap.json      - top-20 features by mean |SHAP value|
    figs/*.png                       - all report figures

Run stages independently (each is safe to re-run on its own once its inputs exist):
    python pipeline.py build_features   # load, merge, clean, engineer features
    python pipeline.py split            # time-based train/test split -> .npy
    python pipeline.py baseline         # class-weighted XGBoost
    python pipeline.py smote            # RandomUnderSampler + SMOTE -> XGBoost
    python pipeline.py evaluate         # ROC/PR curves, threshold tuning, confusion matrices
    python pipeline.py importance       # gain importance + SHAP summary
    python pipeline.py all              # run every stage in order

Note on scale: this was developed and tested in a constrained sandbox (~3.9GB RAM,
1 CPU core). Each stage is written to run as a fresh, short-lived step so memory from
one stage doesn't accumulate into the next (dtypes are aggressively downcast to
float32/int32/int8, and the SMOTE stage undersamples the majority class first --
see the comment in `run_smote()` for why). On a normal machine you can safely
increase `n_estimators`, drop the undersampling cap, or push SMOTE's
`sampling_strategy` closer to 1:1.
"""
import argparse
import gc
import json
import resource
import time

import numpy as np
import pandas as pd


def mem():
    """Peak resident memory (GB) of this process so far."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# =====================================================================
# Stage 1: load, merge, clean, engineer features
# =====================================================================
def run_build_features():
    t0 = time.time()

    def infer_dtype_map(path, sample_rows=2000):
        sample = pd.read_csv(path, nrows=sample_rows)
        dmap = {}
        for col in sample.columns:
            if col in ("TransactionID", "isFraud"):
                continue
            dt = sample[col].dtype
            dmap[col] = "float32" if dt in (np.float64, np.int64) else "category"
        return dmap

    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Loading train_transaction.csv ...", flush=True)
    tx_dmap = infer_dtype_map("data/train_transaction.csv")
    train_tx = pd.read_csv("data/train_transaction.csv", dtype=tx_dmap)
    train_tx["TransactionID"] = train_tx["TransactionID"].astype("int32")
    train_tx["isFraud"] = train_tx["isFraud"].astype("int8")
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] tx shape={train_tx.shape}", flush=True)

    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Loading train_identity.csv ...", flush=True)
    id_dmap = infer_dtype_map("data/train_identity.csv")
    train_id = pd.read_csv("data/train_identity.csv", dtype=id_dmap)
    train_id["TransactionID"] = train_id["TransactionID"].astype("int32")
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] id shape={train_id.shape}", flush=True)

    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Merging ...", flush=True)
    df = train_tx.merge(train_id, on="TransactionID", how="left")
    del train_tx, train_id
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] merged shape={df.shape}", flush=True)

    fraud_rate = df["isFraud"].mean()
    print(f"Fraud rate: {fraud_rate:.4%} (fraud={int(df['isFraud'].sum())}, total={len(df)})", flush=True)

    # Drop columns that are almost entirely missing (>90%)
    missing_frac = df.isna().mean()
    high_missing_cols = missing_frac[missing_frac > 0.90].index.tolist()
    high_missing_cols = [c for c in high_missing_cols if c not in ("isFraud", "TransactionID", "TransactionDT")]
    df = df.drop(columns=high_missing_cols)
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Dropped {len(high_missing_cols)} >90%-missing cols -> {df.shape}", flush=True)

    # Time-based features
    df["Transaction_hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["Transaction_day"] = ((df["TransactionDT"] // (3600 * 24)) % 7).astype("int8")

    # TransactionAmt engineered features (log transform + decimal-cents pattern)
    df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"].astype("float32")).astype("float32")
    df["TransactionAmt_decimal"] = (
        (df["TransactionAmt"] - df["TransactionAmt"].fillna(0).astype("int32")) * 1000
    ).astype("float32")
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Added time/amount features", flush=True)

    # Frequency-encode high-cardinality identifiers
    for col in ["card1", "card2", "addr1", "P_emaildomain"]:
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            df[f"{col}_freq"] = df[col].map(vc).astype("float32")
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Frequency-encoded card1/card2/addr1/P_emaildomain", flush=True)

    # Label-encode categorical columns (NaN -> -1)
    cat_cols = df.select_dtypes(include="category").columns.tolist()
    for col in cat_cols:
        df[col] = df[col].cat.codes.astype("int32")
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Label-encoded {len(cat_cols)} categorical cols", flush=True)

    # Median-impute remaining numeric NaNs
    numeric_cols = [c for c in df.select_dtypes(include=["float32", "float64"]).columns if c != "isFraud"]
    na_counts = df[numeric_cols].isna().sum()
    cols_with_na = na_counts[na_counts > 0].index.tolist()
    medians = {}
    for col in cols_with_na:
        med = df[col].median()
        medians[col] = float(med) if not pd.isna(med) else 0.0
        df[col] = df[col].fillna(medians[col])
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Imputed {len(cols_with_na)} numeric cols with medians", flush=True)

    remaining_na = int(df.isna().sum().sum())
    if remaining_na > 0:
        df = df.fillna(0)
    print(f"Remaining NaNs before fallback fill: {remaining_na}", flush=True)

    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
        elif df[col].dtype == "int64":
            df[col] = df[col].astype("int32")

    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Final shape: {df.shape}", flush=True)
    print(f"Fraud rate: {df['isFraud'].mean():.4%}", flush=True)

    df.to_parquet("data/features_clean.parquet", index=False)
    with open("data/meta.json", "w") as f:
        json.dump(
            {
                "dropped_high_missing_cols": high_missing_cols,
                "n_rows": len(df),
                "n_cols": df.shape[1],
                "fraud_rate": fraud_rate,
            },
            f,
            indent=2,
        )
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Saved features_clean.parquet", flush=True)


# =====================================================================
# Stage 2: time-based train/test split
# =====================================================================
def run_split():
    import pyarrow.parquet as pq

    t0 = time.time()
    print(f"[mem={mem():.2f}GB] Reading parquet ...", flush=True)
    table = pq.read_table("data/features_clean.parquet")
    df = table.to_pandas(self_destruct=True, split_blocks=True)
    del table
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Loaded {df.shape}", flush=True)

    # Sort chronologically and take the last 20% as test -- mirrors real deployment
    # (predicting the future from the past) and avoids leakage a random split can
    # introduce when the same card/device/address appears on both sides of the split.
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("isFraud", "TransactionID", "TransactionDT")]
    split_idx = int(len(df) * 0.8)

    y_all = df["isFraud"].to_numpy()
    X_all = df[feature_cols].to_numpy(dtype="float32")
    del df
    gc.collect()

    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    del X_all, y_all
    gc.collect()

    print(f"Train: {X_train.shape}, fraud rate={y_train.mean():.4%}", flush=True)
    print(f"Test:  {X_test.shape}, fraud rate={y_test.mean():.4%}", flush=True)

    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)
    with open("data/feature_cols.json", "w") as f:
        json.dump(feature_cols, f)
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Saved train/test npy arrays.", flush=True)


# =====================================================================
# Stage 3a: baseline XGBoost, class-weighted (no resampling)
# =====================================================================
def run_baseline():
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    t0 = time.time()
    X_train, y_train = np.load("data/X_train.npy"), np.load("data/y_train.npy")
    X_test, y_test = np.load("data/X_test.npy"), np.load("data/y_test.npy")
    print(f"[mem={mem():.2f}GB] Loaded arrays. X_train={X_train.shape}", flush=True)

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7,
        tree_method="hist", max_bin=128, scale_pos_weight=spw,
        eval_metric="aucpr", n_jobs=1, random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    res = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "scale_pos_weight": float(spw),
    }
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Baseline ROC-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f}", flush=True)

    np.save("data/proba_baseline.npy", proba)
    model.get_booster().save_model("data/model_baseline.json")
    with open("data/results_baseline.json", "w") as f:
        json.dump(res, f, indent=2)


# =====================================================================
# Stage 3b: RandomUnderSampler + SMOTE, then XGBoost
# =====================================================================
def run_smote():
    from xgboost import XGBClassifier
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.metrics import roc_auc_score, average_precision_score

    t0 = time.time()
    X_train, y_train = np.load("data/X_train.npy"), np.load("data/y_train.npy")
    X_test, y_test = np.load("data/X_test.npy"), np.load("data/y_test.npy")
    print(f"[mem={mem():.2f}GB] Loaded arrays. X_train={X_train.shape}", flush=True)

    # Plain SMOTE run on the full ~472K-row training set targeting anywhere near 1:1
    # balance allocates a resampled matrix of hundreds of thousands of rows x 427
    # features -- prohibitive on a memory-constrained machine. We use the standard,
    # documented combination of RandomUnderSampler (trims the majority class first)
    # + SMOTE (oversamples the minority from there) via an imblearn Pipeline. On a
    # bigger machine, raise/remove `under_target` and push `sampling_strategy` in
    # SMOTE closer to 1.0 for a fuller rebalance.
    n_minority = int((y_train == 1).sum())
    under_target = min(100_000, int((y_train == 0).sum()))
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] RandomUnderSampler(majority->{under_target}) "
          f"+ SMOTE(ratio=0.5) ...", flush=True)
    pipeline = ImbPipeline(steps=[
        ("under", RandomUnderSampler(sampling_strategy={0: under_target, 1: n_minority}, random_state=42)),
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)),
    ])
    X_train_sm, y_train_sm = pipeline.fit_resample(X_train, y_train)
    del X_train, y_train
    gc.collect()
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Resampled train={X_train_sm.shape}, "
          f"fraud rate={y_train_sm.mean():.4%}", flush=True)

    model = XGBClassifier(
        n_estimators=250, max_depth=5, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.7,
        tree_method="hist", max_bin=128,
        eval_metric="aucpr", n_jobs=1, random_state=42,
    )
    model.fit(X_train_sm, y_train_sm)
    proba = model.predict_proba(X_test)[:, 1]
    res = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "resampled_train_shape": list(X_train_sm.shape),
        "resampled_fraud_rate": float(y_train_sm.mean()),
        "under_target_majority": under_target,
    }
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] SMOTE model ROC-AUC={res['roc_auc']:.4f} PR-AUC={res['pr_auc']:.4f}", flush=True)

    np.save("data/proba_smote.npy", proba)
    model.get_booster().save_model("data/model_smote.json")
    with open("data/results_smote.json", "w") as f:
        json.dump(res, f, indent=2)


# =====================================================================
# Stage 4: evaluation plots + threshold tuning
# =====================================================================
def run_evaluate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (roc_curve, precision_recall_curve, roc_auc_score,
                                  average_precision_score, confusion_matrix, f1_score,
                                  precision_score, recall_score)

    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    y_test = np.load("data/y_test.npy")
    proba_base = np.load("data/proba_baseline.npy")
    proba_smote = np.load("data/proba_smote.npy")
    with open("data/meta.json") as f:
        meta = json.load(f)

    # -- class imbalance --
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = [int((1 - meta["fraud_rate"]) * meta["n_rows"]), int(meta["fraud_rate"] * meta["n_rows"])]
    bars = ax.bar(["Legitimate", "Fraud"], counts, color=["#4C72B0", "#C44E52"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c, f"{c:,}\n({c/sum(counts):.2%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Transactions")
    ax.set_title(f"Class Imbalance in Training Data (fraud rate = {meta['fraud_rate']:.2%})")
    plt.tight_layout(); plt.savefig("figs/01_class_imbalance.png"); plt.close()

    # -- ROC curves --
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for proba, label, color in [(proba_base, "Baseline (class-weighted)", "#4C72B0"),
                                 (proba_smote, "Undersample+SMOTE", "#C44E52")]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Held-out Test Set"); ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig("figs/02_roc_curve.png"); plt.close()

    # -- PR curves --
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for proba, label, color in [(proba_base, "Baseline (class-weighted)", "#4C72B0"),
                                 (proba_smote, "Undersample+SMOTE", "#C44E52")]:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, label=f"{label} (AP={ap:.3f})", color=color)
    ax.axhline(y_test.mean(), color="gray", ls="--", lw=0.8, label=f"No-skill baseline ({y_test.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Held-out Test Set"); ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig("figs/03_pr_curve.png"); plt.close()

    # -- threshold tuning (SMOTE model) --
    thresholds = np.linspace(0.01, 0.99, 99)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred = (proba_smote >= t).astype(int)
        precisions.append(precision_score(y_test, pred, zero_division=0))
        recalls.append(recall_score(y_test, pred, zero_division=0))
        f1s.append(f1_score(y_test, pred, zero_division=0))
    precisions, recalls, f1s = np.array(precisions), np.array(recalls), np.array(f1s)
    best_f1_idx = int(np.argmax(f1s))
    best_f1_threshold = float(thresholds[best_f1_idx])
    recall_target = 0.90
    valid = np.where(recalls >= recall_target)[0]
    recall_target_threshold = float(thresholds[valid[-1]]) if len(valid) else float(thresholds[0])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(thresholds, precisions, label="Precision", color="#55A868")
    ax.plot(thresholds, recalls, label="Recall", color="#C44E52")
    ax.plot(thresholds, f1s, label="F1", color="#4C72B0")
    ax.axvline(0.5, color="gray", ls=":", lw=1, label="Default 0.50")
    ax.axvline(best_f1_threshold, color="#4C72B0", ls="--", lw=1, label=f"F1-optimal ({best_f1_threshold:.2f})")
    ax.axvline(recall_target_threshold, color="#C44E52", ls="--", lw=1, label=f"90%-recall target ({recall_target_threshold:.2f})")
    ax.set_xlabel("Decision Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold Tuning — SMOTE Model")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(); plt.savefig("figs/04_threshold_tuning.png"); plt.close()

    # -- confusion matrices --
    def plot_cm(ax, y_true, y_pred, title):
        cm = confusion_matrix(y_true, y_pred)
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Legit", "Fraud"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Legit", "Fraud"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
        return cm

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    plot_cm(axes[0], y_test, (proba_smote >= 0.5).astype(int), "Threshold = 0.50 (default)")
    plot_cm(axes[1], y_test, (proba_smote >= best_f1_threshold).astype(int), f"Threshold = {best_f1_threshold:.2f} (F1-optimal)")
    plot_cm(axes[2], y_test, (proba_smote >= recall_target_threshold).astype(int), f"Threshold = {recall_target_threshold:.2f} (90% recall)")
    plt.suptitle("Confusion Matrices — SMOTE Model on Held-out Test Set")
    plt.tight_layout(); plt.savefig("figs/05_confusion_matrices.png"); plt.close()

    def summarize(y_true, proba, t):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        return {"threshold": float(t), "precision": float(precision_score(y_true, pred, zero_division=0)),
                "recall": float(recall_score(y_true, pred, zero_division=0)),
                "f1": float(f1_score(y_true, pred, zero_division=0)),
                "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}

    threshold_summary = {
        "default_0.5": summarize(y_test, proba_smote, 0.5),
        "f1_optimal": summarize(y_test, proba_smote, best_f1_threshold),
        "recall_90pct_target": summarize(y_test, proba_smote, recall_target_threshold),
    }
    with open("data/threshold_summary.json", "w") as f:
        json.dump(threshold_summary, f, indent=2)
    print(json.dumps(threshold_summary, indent=2))
    print("Saved plots to figs/")


# =====================================================================
# Stage 5: feature importance (gain) + SHAP
# =====================================================================
def run_importance():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import xgboost as xgb
    import shap

    t0 = time.time()
    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})
    with open("data/feature_cols.json") as f:
        feature_cols = json.load(f)

    booster = xgb.Booster()
    booster.load_model("data/model_smote.json")
    booster.feature_names = feature_cols

    gain_scores = booster.get_score(importance_type="gain")
    gain_sorted = sorted(gain_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    names = [n for n, _ in gain_sorted][::-1]
    vals = [v for _, v in gain_sorted][::-1]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(names, vals, color="#4C72B0")
    ax.set_xlabel("Average Gain per Split")
    ax.set_title("XGBoost Feature Importance (Gain) — Top 20\nSMOTE Model")
    plt.tight_layout(); plt.savefig("figs/06_feature_importance_gain.png"); plt.close()
    with open("data/top_features_gain.json", "w") as f:
        json.dump(gain_sorted[::-1], f, indent=2)

    X_test = np.load("data/X_test.npy")
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(3000, len(X_test)), replace=False)
    X_sample = X_test[sample_idx]
    del X_test
    gc.collect()

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    fig = plt.figure(figsize=(8, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_cols, max_display=20, show=False)
    plt.title("SHAP Summary — Top 20 Features (SMOTE Model, test sample)")
    plt.tight_layout(); plt.savefig("figs/07_shap_summary.png", dpi=110); plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_rank = sorted(zip(feature_cols, mean_abs_shap.tolist()), key=lambda x: x[1], reverse=True)[:20]
    with open("data/top_features_shap.json", "w") as f:
        json.dump(shap_rank, f, indent=2)
    print(f"[mem={mem():.2f}GB t={time.time()-t0:.0f}s] Done.", flush=True)


STAGES = {
    "build_features": run_build_features,
    "split": run_split,
    "baseline": run_baseline,
    "smote": run_smote,
    "evaluate": run_evaluate,
    "importance": run_importance,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(STAGES.keys()) + ["all"])
    args = parser.parse_args()

    import os
    os.makedirs("data", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    if args.stage == "all":
        for name, fn in STAGES.items():
            print(f"\n{'='*70}\nSTAGE: {name}\n{'='*70}")
            fn()
    else:
        STAGES[args.stage]()
