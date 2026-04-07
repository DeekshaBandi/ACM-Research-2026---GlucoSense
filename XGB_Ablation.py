import pandas as pd
import numpy as np
import os
import itertools
from xgboost import XGBClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, balanced_accuracy_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
# Ensure these paths match your local setup
DATA_PATH = r"c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\MergedDataset.csv"
OUTPUT_DIR = r"c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\XGB_results"
LABEL_COL = "label_cv_numeric"

DROP_COLS = [
    "date", "subject_id", "label_cv_threshold", "label_cv_numeric",
    "Window_Start", "Window_End", "Overlap_Hours", "label"
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ── DATA LOADING ─────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
participants = sorted(df['subject_id'].unique())

def get_sensor_groups(columns):
    return {
        "Cardiac": [c for c in columns if any(pre in c for pre in ['HR', 'IBI', 'BVP'])],
        "ACC": [c for c in columns if 'ACC' in c],
        "EDA": [c for c in columns if 'EDA' in c],
        "TEMP": [c for c in columns if 'TEMP' in c]
    }

sensor_map = get_sensor_groups([c for c in df.columns if c not in DROP_COLS])
modalities = list(sensor_map.keys())

# ── ABLATION LOOP ───────────────────────────────────────────────────────────
master_results = []

for r in range(1, 5):
    for combo in itertools.combinations(modalities, r):
        combo_name = "-".join(combo)
        current_features = []
        for m in combo:
            current_features.extend(sensor_map[m])
            
        print(f"\nEvaluating Combination: {combo_name}")
        
        X = df[current_features].values
        y = df[LABEL_COL].values
        groups = df["subject_id"].values
        
        # Lists to store metrics for each LOSO fold
        fold_metrics = {
            "f1": [], "auroc": [], "bal_acc": [], "sens": [], "spec": []
        }

        for test_pid in participants:
            train_mask = groups != test_pid
            test_mask  = groups == test_pid

            X_train, y_train = X[train_mask], y[train_mask]
            X_test,  y_test  = X[test_mask],  y[test_mask]

            if len(np.unique(y_test)) < 2:
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

            # Calculate metrics for this specific participant
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            
            fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            fold_metrics["auroc"].append(roc_auc_score(y_test, y_proba))
            fold_metrics["bal_acc"].append(balanced_accuracy_score(y_test, y_pred))
            fold_metrics["sens"].append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fold_metrics["spec"].append(tn / (tn + fp) if (tn + fp) > 0 else 0)

        # Average the metrics across all participants
        master_results.append({
            "Combination": combo_name,
            "Sensors": ", ".join(combo),
            "Mean_F1": np.mean(fold_metrics["f1"]),
            "Mean_AUROC": np.mean(fold_metrics["auroc"]),
            "Mean_Bal_Acc": np.mean(fold_metrics["bal_acc"]),
            "Mean_Sensitivity": np.mean(fold_metrics["sens"]),
            "Mean_Specificity": np.mean(fold_metrics["spec"])
        })
        print(f"  -> F1: {master_results[-1]['Mean_F1']:.3f} | AUROC: {master_results[-1]['Mean_AUROC']:.3f}")

# ── SAVE SUMMARY ────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(master_results).sort_values(by="Mean_F1", ascending=False)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "XGB_Ablation_Full_Metrics.csv"), index=False)
print(f"\nAblation Complete. Detailed summary saved to {OUTPUT_DIR}")