"""
Baseline model training — all features, LOSO-CV
-------------------------------------------------
Role of this script
  Provides a FULL-FEATURE-SET baseline comparison across three classifiers.
  Logistic Regression is deliberately scoped to this baseline only and is
  NOT included in the sensor-ablation loop (see ablation.py). RF and XGB
  are the co-primary ablation models; LR is retained here as a linear-
  interpretable reference on the complete feature set.

Models:
  - Logistic Regression (L2, balanced class weight)   [baseline-only]
  - Random Forest       (100 trees, balanced class weight)  [co-primary]
  - XGBoost             (100 trees, scale_pos_weight for balance)  [co-primary]

Metrics reported per model:
  AUROC, balanced accuracy, F1 (macro), sensitivity, specificity
  + per-fold AUROC for transparency about variance

Outputs:
  results/baseline_predictions_{model}.csv  — per-row predictions
  results/baseline_metrics.csv              — aggregate metric table
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_config import ALL_FEATURES, LABEL_COL
from modeling_utils  import load_data, prepare_matrix, run_loso_cv, compute_fold_metrics

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
df = load_data()
X, y, parts = prepare_matrix(df, ALL_FEATURES, LABEL_COL)

n_pos   = int(y.sum())
n_neg   = len(y) - n_pos

print(f"Dataset: {len(y)} samples | {n_pos} high / {n_neg} low | "
      f"{len(set(parts))} participants\n")

# ── Models ────────────────────────────────────────────────────────────────────
# Note: XGBoost scale_pos_weight is computed per LOSO fold inside run_loso_cv()
# to avoid leaking test-set class distribution into training.
MODELS = {
    "logistic_regression": LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=1,  # overridden per-fold by run_loso_cv
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}

# Logistic regression needs scaling; tree models do not
SCALE = {
    "logistic_regression": True,
    "random_forest":       False,
    "xgboost":             False,
}

# ── Training loop ─────────────────────────────────────────────────────────────

all_metrics = []

for model_name, clf in MODELS.items():
    print(f"── {model_name.upper()} ──────────────────────────────")
    scale = SCALE[model_name]

    preds = run_loso_cv(clf, X, y, parts, scale=scale)
    preds.to_csv(f"results/baseline_predictions_{model_name}.csv", index=False)

    m = compute_fold_metrics(preds)

    print(f"  AUROC (fold)  : {m['auroc_fold_mean']:.3f} ± {m['auroc_fold_std']:.3f}  "
          f"(pooled {m['auroc_pooled']:.3f}, 95% CI [{m['auroc_ci_lo']:.3f}, {m['auroc_ci_hi']:.3f}])")
    print(f"  BalAcc (fold) : {m['bal_acc_fold_mean']:.3f} ± {m['bal_acc_fold_std']:.3f}  "
          f"(pooled {m['bal_acc_pooled']:.3f})")
    print(f"  F1 mac (fold) : {m['f1_fold_mean']:.3f} ± {m['f1_fold_std']:.3f}  "
          f"(pooled {m['f1_pooled']:.3f})")
    print(f"  Sens   (fold) : {m['sens_fold_mean']:.3f} ± {m['sens_fold_std']:.3f}  "
          f"(pooled {m['sens_pooled']:.3f})")
    print(f"  Spec   (fold) : {m['spec_fold_mean']:.3f} ± {m['spec_fold_std']:.3f}  "
          f"(pooled {m['spec_pooled']:.3f})")
    print(f"  AUROC folds evaluable: {m['auroc_folds_eval']}/{len(set(parts))}")
    print()

    all_metrics.append({"model": model_name, **m})

# ── Summary table ─────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv("results/baseline_metrics.csv", index=False)

print("══════════════════════════════════════════════════════════════")
print("SUMMARY — all features, LOSO-CV (fold means ± std)")
print("══════════════════════════════════════════════════════════════")
display_cols = ["model", "auroc_fold_mean", "auroc_fold_std", "auroc_ci_lo", "auroc_ci_hi",
                "bal_acc_fold_mean", "f1_fold_mean", "sens_fold_mean", "spec_fold_mean"]
print(metrics_df[display_cols].to_string(index=False))
print(f"\nSaved → results/baseline_metrics.csv")
print(f"Saved → results/baseline_predictions_{{model}}.csv")
