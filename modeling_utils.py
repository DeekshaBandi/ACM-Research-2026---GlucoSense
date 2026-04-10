"""
Modeling utilities: data loading, matrix preparation, and scaling.

Key design decisions
--------------------
Missing values
  5 participant-days have valid IBI but NaN for all other modalities (HR, ACC,
  EDA, TEMP) — these are partial-wear days that passed the 100-beat IBI
  threshold but not the 6-hour coverage threshold for other signals.
  Strategy: per-subset, drop any row missing a feature in that subset.
  This preserves all 117 rows for IBI-only models and uses 112 rows for
  all other subsets.  Imputation is intentionally avoided — fabricating
  values for days where the device was barely worn would introduce noise.

Scaling
  StandardScaler is applied INSIDE each LOSO-CV fold:
    scaler.fit(X_train) → scaler.transform(X_train), scaler.transform(X_test)
  This prevents any test-set statistics leaking into the training pipeline.
  Tree-based models (Random Forest) do not require scaling; Logistic Regression
  and SVM do. Both paths are supported via the `scale` argument.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/wearable_features_labeled.csv"


def load_data() -> pd.DataFrame:
    """Load the labeled feature table and normalise types."""
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["participant"] = df["participant"].astype(str).str.zfill(3)
    return df


def prepare_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str = "label",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract X, y, and participant IDs for a given feature subset.

    Rows missing any feature in `feature_cols` are dropped (not imputed).
    Returns arrays aligned by index.

    Parameters
    ----------
    df           : full labeled dataframe from load_data()
    feature_cols : list of column names to use as features
    label_col    : target column (default 'label' == label_median)

    Returns
    -------
    X            : (n_samples, n_features) float array
    y            : (n_samples,) int array  {0=low, 1=high}
    participants : (n_samples,) str array  for LOSO-CV grouping
    """
    subset = df[["participant"] + feature_cols + [label_col]].dropna()
    X = subset[feature_cols].values.astype(float)
    y = subset[label_col].values.astype(int)
    participants = subset["participant"].values
    return X, y, participants


def get_loso_splits(participants: np.ndarray):
    """
    Yield (train_idx, test_idx, held_out_participant) for LOSO-CV.
    Each fold holds out all days from one participant.
    """
    unique = sorted(set(participants))
    for pid in unique:
        test_mask  = participants == pid
        train_mask = ~test_mask
        yield np.where(train_mask)[0], np.where(test_mask)[0], pid


def scale_fold(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fit StandardScaler on X_train, apply to both.
    Returns (X_train_scaled, X_test_scaled).
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    return X_train_s, X_test_s


# ── LOSO-CV runner ───────────────────────────────────────────────────────────

def run_loso_cv(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    participants: np.ndarray,
    scale: bool = True,
) -> pd.DataFrame:
    """
    Run Leave-One-Subject-Out Cross-Validation.

    Parameters
    ----------
    clf          : sklearn-compatible estimator (must have fit/predict/predict_proba)
    X            : feature matrix (n_samples, n_features)
    y            : binary labels (n_samples,)
    participants : participant ID per sample (n_samples,)
    scale        : if True, StandardScaler is fit on train and applied to both
                   train and test within each fold

    Returns
    -------
    DataFrame with one row per test sample:
        participant, fold, y_true, y_pred, y_prob
    """
    records = []

    for fold_i, (tr_idx, te_idx, pid) in enumerate(get_loso_splits(participants)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        if scale:
            X_tr, X_te = scale_fold(X_tr, X_te)

        # Recompute class-balance weight from training fold only (prevents leakage)
        if hasattr(clf, "scale_pos_weight"):
            n_pos_tr = int(y_tr.sum())
            n_neg_tr = len(y_tr) - n_pos_tr
            clf.set_params(scale_pos_weight=n_neg_tr / n_pos_tr)

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        # Probability of class 1 (needed for AUROC)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        elif hasattr(clf, "decision_function"):
            raw = clf.decision_function(X_te)
            # Squash to [0, 1] via min-max for a usable probability proxy
            r_min, r_max = raw.min(), raw.max()
            y_prob = (raw - r_min) / (r_max - r_min + 1e-9)
        else:
            y_prob = y_pred.astype(float)

        for j in range(len(te_idx)):
            records.append({
                "participant": pid,
                "fold":        fold_i + 1,
                "y_true":      int(y_te[j]),
                "y_pred":      int(y_pred[j]),
                "y_prob":      float(y_prob[j]),
            })

    return pd.DataFrame(records)


# ── Metric computation ──────────────────────────────────────────────────────

def compute_fold_metrics(preds: pd.DataFrame, n_boot: int = 1000) -> dict:
    """
    Compute evaluation metrics per LOSO fold (mean ± std) and pooled.

    Per-fold aggregation gives equal weight to each participant, which is
    the correct approach for LOSO-CV.  Pooled metrics are retained as
    secondary for reference.

    Parameters
    ----------
    preds  : DataFrame with columns participant, y_true, y_pred, y_prob
    n_boot : bootstrap resamples for AUROC 95% CI

    Returns
    -------
    dict with keys for each of the 5 metrics (AUROC, balanced accuracy,
    F1 macro, sensitivity, specificity):
        {metric}_fold_mean, {metric}_fold_std, {metric}_pooled
    Plus: auroc_folds_eval, auroc_ci_lo, auroc_ci_hi, n, n_high, n_low
    """
    from sklearn.metrics import (
        roc_auc_score, balanced_accuracy_score,
        f1_score, confusion_matrix,
    )

    yt_all  = preds["y_true"].values
    yp_all  = preds["y_pred"].values
    ypr_all = preds["y_prob"].values

    # ── Per-fold metrics ──────────────────────────────────────────────
    fold_auroc, fold_ba, fold_f1, fold_sens, fold_spec = [], [], [], [], []

    for _, g in preds.groupby("participant"):
        yt  = g["y_true"].values
        yp  = g["y_pred"].values
        ypr = g["y_prob"].values

        # AUROC requires both classes in the fold
        if len(np.unique(yt)) == 2:
            fold_auroc.append(roc_auc_score(yt, ypr))

        fold_ba.append(balanced_accuracy_score(yt, yp))
        fold_f1.append(f1_score(yt, yp, average="macro", zero_division=0))

        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        fold_sens.append(tp / (tp + fn) if (tp + fn) > 0 else np.nan)
        fold_spec.append(tn / (tn + fp) if (tn + fp) > 0 else np.nan)

    def _nanstats(vals):
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            return np.nan, np.nan
        return float(np.mean(clean)), float(np.std(clean))

    auroc_fm, auroc_fs = (float(np.mean(fold_auroc)), float(np.std(fold_auroc))) if fold_auroc else (np.nan, np.nan)
    ba_fm, ba_fs       = _nanstats(fold_ba)
    f1_fm, f1_fs       = _nanstats(fold_f1)
    sens_fm, sens_fs   = _nanstats(fold_sens)
    spec_fm, spec_fs   = _nanstats(fold_spec)

    # ── Pooled metrics ────────────────────────────────────────────────
    try:
        auroc_p = roc_auc_score(yt_all, ypr_all)
    except ValueError:
        auroc_p = np.nan

    ba_p = balanced_accuracy_score(yt_all, yp_all)
    f1_p = f1_score(yt_all, yp_all, average="macro", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yt_all, yp_all, labels=[0, 1]).ravel()
    sens_p = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec_p = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    # ── Bootstrap 95% CI for pooled AUROC ─────────────────────────────
    rng  = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yt_all), len(yt_all))
        if len(np.unique(yt_all[idx])) == 2:
            boot.append(roc_auc_score(yt_all[idx], ypr_all[idx]))
    ci_lo, ci_hi = (np.percentile(boot, [2.5, 97.5]) if boot
                    else (np.nan, np.nan))

    r = lambda x: round(x, 3) if not np.isnan(x) else np.nan

    return {
        "n":                 len(yt_all),
        "n_high":            int(yt_all.sum()),
        "n_low":             int(len(yt_all) - yt_all.sum()),
        "auroc_fold_mean":   r(auroc_fm),
        "auroc_fold_std":    r(auroc_fs),
        "auroc_folds_eval":  len(fold_auroc),
        "auroc_pooled":      r(auroc_p),
        "auroc_ci_lo":       r(ci_lo),
        "auroc_ci_hi":       r(ci_hi),
        "bal_acc_fold_mean": r(ba_fm),
        "bal_acc_fold_std":  r(ba_fs),
        "bal_acc_pooled":    r(ba_p),
        "f1_fold_mean":      r(f1_fm),
        "f1_fold_std":       r(f1_fs),
        "f1_pooled":         r(f1_p),
        "sens_fold_mean":    r(sens_fm),
        "sens_fold_std":     r(sens_fs),
        "sens_pooled":       r(sens_p),
        "spec_fold_mean":    r(spec_fm),
        "spec_fold_std":     r(spec_fs),
        "spec_pooled":       r(spec_p),
    }


# ── Self-test / inspection ────────────────────────────────────────────────────
if __name__ == "__main__":
    from feature_config import ALL_FEATURES, ABLATION_SUBSETS, LABEL_COL

    df = load_data()

    # ── Sample attrition summary ─────────────────────────────────────────
    cgm_path = DATA_PATH.replace("wearable_features_labeled", "../data/cgm_daily_labels")
    try:
        n_cgm = len(pd.read_csv("data/cgm_daily_labels.csv"))
    except FileNotFoundError:
        n_cgm = "?"
    n_joined    = len(df)
    ibi_feats   = [c for c in df.columns if c.startswith("ibi_")]
    n_ibi_only  = len(df.dropna(subset=ibi_feats))
    non_ibi     = [c for c in ALL_FEATURES if not c.startswith("ibi_")]
    n_non_ibi   = len(df.dropna(subset=non_ibi))

    print("── Sample attrition ──────────────────────────────────────────")
    print(f"  CGM participant-days (after quality filter) : {n_cgm}")
    print(f"  After wearable join                         : {n_joined}")
    print(f"  IBI-only subsets (rows with valid IBI)      : {n_ibi_only}")
    print(f"  Non-IBI subsets (rows with all modalities)  : {n_non_ibi}")
    print(f"  Participants                                : {df['participant'].nunique()}")
    print()

    # Show matrix stats for every ablation subset
    print(f"{'Subset':<22s}  {'rows':>5s}  {'feats':>5s}  {'high':>5s}  {'low':>5s}  {'balance':>8s}")
    print("-" * 65)
    for name, feats in ABLATION_SUBSETS.items():
        X, y, parts = prepare_matrix(df, feats, LABEL_COL)
        n_high = y.sum()
        n_low  = len(y) - n_high
        balance = f"{n_high/len(y)*100:.1f}% / {n_low/len(y)*100:.1f}%"
        print(f"  {name:<20s}  {len(y):>5d}  {len(feats):>5d}  {n_high:>5d}  {n_low:>5d}  {balance:>10s}")

    # Demonstrate per-fold scaling
    print("\n--- Scaling demo (first LOSO fold of 'all' subset) ---")
    X, y, parts = prepare_matrix(df, ALL_FEATURES, LABEL_COL)
    for tr_idx, te_idx, pid in get_loso_splits(parts):
        X_tr_s, X_te_s = scale_fold(X[tr_idx], X[te_idx])
        print(f"Held-out participant : {pid}")
        print(f"Train : {len(tr_idx)} rows  |  Test : {len(te_idx)} rows")
        print(f"Train mean (post-scale) : {X_tr_s.mean(axis=0).round(4)[:4]} ...")
        print(f"Train std  (post-scale) : {X_tr_s.std(axis=0).round(4)[:4]} ...")
        print(f"Test  mean (post-scale) : {X_te_s.mean(axis=0).round(4)[:4]} ...")
        break  # just show the first fold
