# baseline_xgboost_dataset1.py
# Stage 2 — Baseline XGBoost | Dataset 1
# LOSO-CV | Binary classification of high vs low glycemic variability

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, recall_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Projects\glucosense_data\merged_de_with_wearables.csv"
LABEL_COL = "label"
DROP_COLS = [
    "participant", "date", "label", "label_str",
    "label_median", "label_p75", "label_clinical",
    "cv_threshold_median", "cv_threshold_p75",
    # drop raw CGM cols so model only sees wearable features
    "n_readings", "mean_glucose", "sd_glucose", "cv",
    "tir_70_180", "tar_180", "tbr_70", "mage"
]
# ────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Participants: {sorted(df['participant'].unique())}")
print(f"Label distribution:\n{df[LABEL_COL].value_counts()}\n")

# Build feature matrix — keep only wearable feature columns
feature_cols = [c for c in df.columns if c not in DROP_COLS]
print(f"Features used ({len(feature_cols)}): {feature_cols}\n")

X = df[feature_cols].values
y = df[LABEL_COL].values
groups = df["participant"].values

# ── LOSO-CV ─────────────────────────────────────────────────────────────────
participants = sorted(df["participant"].unique())

results = []

for test_pid in participants:
    train_mask = groups != test_pid
    test_mask  = groups == test_pid

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    if len(np.unique(y_test)) < 2:
        print(f"  Participant {test_pid}: skipped (only one class in test set)")
        continue

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    results.append({
        "participant":       test_pid,
        "n_test_days":       len(y_test),
        "auroc":             roc_auc_score(y_test, y_proba),
        "balanced_acc":      balanced_accuracy_score(y_test, y_pred),
        "f1":                f1_score(y_test, y_pred, zero_division=0),
        "sensitivity":       sensitivity,
        "specificity":       specificity,
    })
    print(f"  Participant {test_pid:>2} | AUROC={results[-1]['auroc']:.3f} | "
          f"BalAcc={results[-1]['balanced_acc']:.3f} | F1={results[-1]['f1']:.3f} | "
          f"Sens={sensitivity:.3f} | Spec={specificity:.3f}")

# ── SUMMARY ─────────────────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
print("\n── LOSO-CV Summary (Dataset 1) ──────────────────────────────────")
print(f"  N subjects evaluated : {len(res_df)}")
for metric in ["auroc", "balanced_acc", "f1", "sensitivity", "specificity"]:
    vals = res_df[metric].dropna()
    print(f"  {metric:<16} mean={vals.mean():.3f}  std={vals.std():.3f}  "
          f"[{vals.min():.3f} – {vals.max():.3f}]")

res_df.to_csv("results_xgboost_dataset1.csv", index=False)
print("\nPer-subject results saved to results_xgboost_dataset1.csv")