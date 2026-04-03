# sensor_ablation.py
# Stage 3 — Sensor Ablation with Random Forest | Dataset 1 (merged_de)
# LOSO-CV | Binary classification of high vs low glycemic variability

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score,
    f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_PATH = r"C:\Projects\glucosense_data\merged_de_with_wearables.csv"

DROP_ALWAYS = [
    "participant", "date", "label", "label_str",
    "label_median", "label_p75", "label_clinical",
    "cv_threshold_median", "cv_threshold_p75",
    "n_readings", "mean_glucose", "sd_glucose", "cv",
    "tir_70_180", "tar_180", "tbr_70", "mage"
]

SENSOR_SUBSETS = {
    "HR only":         ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_median"],
    "IBI only":        ["ibi_mean", "ibi_std", "ibi_rmssd", "ibi_pnn50"],
    "EDA only":        ["eda_mean", "eda_std", "eda_min", "eda_max"],
    "TEMP only":       ["temp_mean", "temp_std", "temp_min", "temp_max"],
    "ACC only":        ["acc_mean", "acc_std", "acc_min", "acc_max"],
    "BVP only":        ["bvp_mean", "bvp_std", "bvp_min", "bvp_max"],
    "HR + IBI":        ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_median",
                        "ibi_mean", "ibi_std", "ibi_rmssd", "ibi_pnn50"],
    "HR + EDA":        ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_median",
                        "eda_mean", "eda_std", "eda_min", "eda_max"],
    "HR + ACC":        ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_median",
                        "acc_mean", "acc_std", "acc_min", "acc_max"],
    "IBI + EDA":       ["ibi_mean", "ibi_std", "ibi_rmssd", "ibi_pnn50",
                        "eda_mean", "eda_std", "eda_min", "eda_max"],
    "HR + IBI + EDA":  ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_median",
                        "ibi_mean", "ibi_std", "ibi_rmssd", "ibi_pnn50",
                        "eda_mean", "eda_std", "eda_min", "eda_max"],
    "HR + IBI + ACC":  ["hr_mean", "hr_std", "hr_min", "hr_max", "hr_median",
                        "ibi_mean", "ibi_std", "ibi_rmssd", "ibi_pnn50",
                        "acc_mean", "acc_std", "acc_min", "acc_max"],
    "All sensors":     None  # None = use everything not in DROP_ALWAYS
}
# ─────────────────────────────────────────────────────────────────────────

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")

all_feature_cols = [c for c in df.columns if c not in DROP_ALWAYS]
groups = df["participant"].values
y_all  = df["label"].values
participants = sorted(df["participant"].unique())


def run_loso(feature_cols):
    X = df[feature_cols].values
    y = y_all
    results = []

    for test_pid in participants:
        train_mask = groups != test_pid
        test_mask  = groups == test_pid

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(np.unique(y_test)) < 2:
            continue

        neg = np.sum(y_train == 0)
        pos = np.sum(y_train == 1)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            class_weight="balanced",
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        results.append({
            "auroc":        roc_auc_score(y_test, y_proba),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "f1":           f1_score(y_test, y_pred, zero_division=0),
            "sensitivity":  sensitivity,
            "specificity":  specificity,
        })

    res = pd.DataFrame(results)
    return {
        "n_evaluated": len(res),
        "auroc":        res["auroc"].mean(),
        "balanced_acc": res["balanced_acc"].mean(),
        "f1":           res["f1"].mean(),
        "sensitivity":  res["sensitivity"].mean(),
        "specificity":  res["specificity"].mean(),
    }


# ── RUN ALL SUBSETS ──────────────────────────────────────────────────────
summary_rows = []

for subset_name, feature_list in SENSOR_SUBSETS.items():
    if feature_list is None:
        feature_list = all_feature_cols

    # only keep features that actually exist in the dataframe
    available = [f for f in feature_list if f in df.columns]
    missing   = [f for f in feature_list if f not in df.columns]

    if missing:
        print(f"  [!] {subset_name}: missing columns {missing}, skipping")
        continue

    print(f"  Running: {subset_name} ({len(available)} features)...")
    metrics = run_loso(available)
    metrics["subset"] = subset_name
    metrics["n_features"] = len(available)
    summary_rows.append(metrics)

# ── PRINT SUMMARY TABLE ──────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)[
    ["subset", "n_features", "auroc", "balanced_acc", "f1", "sensitivity", "specificity"]
].sort_values("f1", ascending=False)

print("\n── Sensor Ablation Results (Random Forest | LOSO-CV) ────────────────")
print(summary_df.to_string(index=False, float_format="{:.3f}".format))

summary_df.to_csv("results_ablation_de.csv", index=False)
print("\nSaved to results_ablation_de.csv")