"""
Sensor-Ablation Random Forest Study
====================================
Trains RF classifiers on different wearable sensor subsets using LOSO-CV
on model_ready_dataset.csv (wearable features + CGM-derived labels).

Sensor groups:
  - HR/IBI  (HR_*, IBI_*, BVP_*)
  - ACC     (ACC_*)
  - EDA     (EDA_*)
  - TEMP    (TEMP_*)

Ablation combos: all singles, all pairs, and all-sensors combined.

Metrics: AUROC, balanced accuracy, F1, sensitivity, specificity
         with 95% bootstrap confidence intervals.
"""

import os
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, f1_score,
    recall_score, confusion_matrix,
)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "output", "model_ready_dataset.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output", "RF_ablation_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Sensor groups ---
SENSOR_GROUPS = {
    "HR_IBI": ["HR_mean", "HR_std", "HR_90th", "HR_10th",
               "IBI_mean", "IBI_std", "IBI_90th", "IBI_10th", "IBI_rmssd",
               "BVP_mean", "BVP_std", "BVP_90th", "BVP_10th"],
    "ACC":    ["ACC_mean", "ACC_std", "ACC_90th", "ACC_10th"],
    "EDA":    ["EDA_mean", "EDA_std", "EDA_90th", "EDA_10th"],
    "TEMP":   ["TEMP_mean", "TEMP_std", "TEMP_90th", "TEMP_10th"],
}

NON_FEATURE_COLS = ["subject_id", "date", "label", "label_str"]


def compute_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def bootstrap_ci(values, n_boot=2000, ci=95, rng_seed=42):
    """Bootstrap 95% CI for a list of per-fold metric values."""
    rng = np.random.RandomState(rng_seed)
    values = np.array(values)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True))
                  for _ in range(n_boot)]
    lo = np.percentile(boot_means, (100 - ci) / 2)
    hi = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return lo, hi


def run_loso(X, y, groups):
    """Run LOSO-CV and return per-fold metrics."""
    logo = LeaveOneGroupOut()
    fold_metrics = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)

        # AUROC needs both classes or at least probabilities
        if len(np.unique(y_test)) == 1:
            auroc = np.nan
        else:
            auroc = roc_auc_score(y_test, y_prob[:, 1])

        fold_metrics.append({
            "auroc": auroc,
            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "sensitivity": recall_score(y_test, y_pred, zero_division=0),
            "specificity": compute_specificity(y_test, y_pred),
        })

    return fold_metrics


def summarise(fold_metrics):
    """Aggregate per-fold metrics into mean + 95% CI."""
    summary = {}
    for metric in ["auroc", "balanced_acc", "f1", "sensitivity", "specificity"]:
        vals = [f[metric] for f in fold_metrics if not np.isnan(f[metric])]
        if len(vals) == 0:
            summary[metric] = np.nan
            summary[f"{metric}_ci_lo"] = np.nan
            summary[f"{metric}_ci_hi"] = np.nan
        else:
            summary[metric] = np.mean(vals)
            lo, hi = bootstrap_ci(vals)
            summary[f"{metric}_ci_lo"] = lo
            summary[f"{metric}_ci_hi"] = hi
    return summary


def main():
    df = pd.read_csv(DATA_PATH)
    y = df["label"].values
    groups = df["subject_id"].values

    # Verify all expected features exist
    all_features = [c for g in SENSOR_GROUPS.values() for c in g]
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    modalities = list(SENSOR_GROUPS.keys())
    results = []

    # --- Build ablation combos: singles, pairs, and all ---
    combos = []
    for r in range(1, len(modalities) + 1):
        combos.extend(itertools.combinations(modalities, r))

    print(f"Dataset: {len(df)} rows, {df['subject_id'].nunique()} participants")
    print(f"Label distribution: {dict(df['label_str'].value_counts())}")
    print(f"Running {len(combos)} sensor combinations with LOSO-CV\n")

    for combo in combos:
        combo_name = " + ".join(combo)
        features = []
        for m in combo:
            features.extend(SENSOR_GROUPS[m])

        X = df[features].values
        fold_metrics = run_loso(X, y, groups)
        summary = summarise(fold_metrics)

        row = {"sensor_combo": combo_name, "n_features": len(features)}
        row.update(summary)
        results.append(row)

        auroc_str = f"{summary['auroc']:.3f} [{summary['auroc_ci_lo']:.3f}-{summary['auroc_ci_hi']:.3f}]"
        bal_acc_str = f"{summary['balanced_acc']:.3f} [{summary['balanced_acc_ci_lo']:.3f}-{summary['balanced_acc_ci_hi']:.3f}]"
        print(f"  {combo_name:<30s}  AUROC={auroc_str}  BalAcc={bal_acc_str}")

    # --- Save results ---
    results_df = pd.DataFrame(results).sort_values("auroc", ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "ablation_summary.csv"), index=False)

    # --- Print ranked summary ---
    print("\n" + "=" * 70)
    print("RANKED RESULTS (by AUROC)")
    print("=" * 70)
    for _, row in results_df.iterrows():
        print(f"  {row['sensor_combo']:<30s}  "
              f"AUROC={row['auroc']:.3f}  "
              f"BalAcc={row['balanced_acc']:.3f}  "
              f"F1={row['f1']:.3f}  "
              f"Sens={row['sensitivity']:.3f}  "
              f"Spec={row['specificity']:.3f}")

    print(f"\nResults saved to {OUTPUT_DIR}/ablation_summary.csv")


if __name__ == "__main__":
    main()
