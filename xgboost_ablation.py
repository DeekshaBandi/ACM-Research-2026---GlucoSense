"""
XGBoost Sensor Ablation — GlucoSense (Laasya RF Branch)
========================================================
Trains an XGBoost binary classifier on MergedDataset.csv across a baseline
condition and multiple wearable sensor subsets.

NO CGM data is used. All features come from the wearable sensors only.

Evaluation: Leave-One-Subject-Out Cross-Validation (LOSOCV) with SMOTE
applied to each training fold to handle the class imbalance (82 low / 52 high).

Usage:
    python xgboost_ablation.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import LeaveOneGroupOut
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────
# SENSOR FEATURE GROUPS
# All columns present in MergedDataset.csv, grouped by modality.
# No CGM columns exist in this dataset — nothing to exclude.
# ─────────────────────────────────────────────────────────────
SENSOR_GROUPS = {
    "HR":   ["HR_mean",  "HR_std",  "HR_90th",  "HR_10th"],
    "IBI":  ["IBI_mean", "IBI_std", "IBI_90th", "IBI_10th", "IBI_rmssd"],
    "EDA":  ["EDA_mean", "EDA_std", "EDA_90th", "EDA_10th"],
    "ACC":  ["ACC_mean", "ACC_std", "ACC_90th", "ACC_10th"],
    "TEMP": ["TEMP_mean","TEMP_std","TEMP_90th","TEMP_10th"],
    "BVP":  ["BVP_mean", "BVP_std", "BVP_90th", "BVP_10th"],
}

# Ablation configs: name → list of modality keys to include
ABLATION_CONFIGS = {
    "Baseline (majority-class dummy)":  None,          # DummyClassifier
    "HR only":                          ["HR"],
    "IBI only":                         ["IBI"],
    "EDA only":                         ["EDA"],
    "ACC only":                         ["ACC"],
    "TEMP only":                        ["TEMP"],
    "BVP only":                         ["BVP"],
    "HR + IBI":                         ["HR", "IBI"],
    "HR + ACC":                         ["HR", "ACC"],
    "HR + EDA":                         ["HR", "EDA"],
    "HR + IBI + ACC":                   ["HR", "IBI", "ACC"],
    "HR + IBI + EDA":                   ["HR", "IBI", "EDA"],
    "All wearable modalities":          ["HR", "IBI", "EDA", "ACC", "TEMP", "BVP"],
}

XGB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

RANDOM_STATE = 42


def run_losocv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, use_dummy: bool = False):
    """LOSOCV with SMOTE on each training fold. Returns aggregated metrics."""
    logo = LeaveOneGroupOut()
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=min(3, np.bincount(y).min() - 1))

    all_true, all_pred, all_proba = [], [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if use_dummy:
            clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            probas = clf.predict_proba(X_te)[:, 1]
        else:
            # Only apply SMOTE if minority class has enough samples
            if np.bincount(y_tr).min() >= smote.k_neighbors + 1:
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

            clf = XGBClassifier(**XGB_PARAMS)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            probas = clf.predict_proba(X_te)[:, 1]

        all_true.extend(y_te)
        all_pred.extend(preds)
        all_proba.extend(probas)

    all_true  = np.array(all_true)
    all_pred  = np.array(all_pred)
    all_proba = np.array(all_proba)

    return {
        "Accuracy":  round(accuracy_score(all_true, all_pred), 4),
        "Precision": round(precision_score(all_true, all_pred, zero_division=0), 4),
        "Recall":    round(recall_score(all_true, all_pred, zero_division=0), 4),
        "F1":        round(f1_score(all_true, all_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(all_true, all_proba), 4),
        "all_true":  all_true,
        "all_pred":  all_pred,
    }


def main():
    data_path = Path("MergedDataset.csv")
    df = pd.read_csv(data_path)

    print(f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Classes: {df['label_cv_threshold'].value_counts().to_dict()}")
    print(f"Subjects: {df['subject_id'].nunique()}\n")

    groups = df["subject_id"].values
    y = df["label_cv_numeric"].values  # 1=high, 0=low

    records = []

    for config_name, modality_keys in ABLATION_CONFIGS.items():
        is_dummy = modality_keys is None

        if is_dummy:
            # Dummy needs *some* feature array; use a zeros placeholder
            X = np.zeros((len(df), 1))
            n_feat = 0
        else:
            feat_cols = []
            for key in modality_keys:
                feat_cols += [c for c in SENSOR_GROUPS[key] if c in df.columns]
            X = df[feat_cols].fillna(df[feat_cols].median()).values
            n_feat = len(feat_cols)

        metrics = run_losocv(X, y, groups, use_dummy=is_dummy)

        row = {
            "Configuration": config_name,
            "# Features": n_feat,
            "Accuracy":  metrics["Accuracy"],
            "Precision": metrics["Precision"],
            "Recall":    metrics["Recall"],
            "F1":        metrics["F1"],
            "ROC-AUC":   metrics["ROC-AUC"],
        }
        records.append(row)

        print(f"[{config_name}]")
        print(
            f"  Acc={metrics['Accuracy']:.4f}  "
            f"Prec={metrics['Precision']:.4f}  "
            f"Recall={metrics['Recall']:.4f}  "
            f"F1={metrics['F1']:.4f}  "
            f"AUC={metrics['ROC-AUC']:.4f}  "
            f"(n_feat={n_feat})"
        )

        # Per-class breakdown for key configs
        if config_name in ("Baseline (majority-class dummy)", "All wearable modalities"):
            print(classification_report(
                metrics["all_true"], metrics["all_pred"],
                target_names=["low (0)", "high (1)"], zero_division=0
            ))

    # ── Summary table ────────────────────────────────────────
    results_df = pd.DataFrame(records).sort_values("F1", ascending=False)

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (sorted by F1)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    out_path = Path("results/xgboost_ablation_results.csv")
    out_path.parent.mkdir(exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
