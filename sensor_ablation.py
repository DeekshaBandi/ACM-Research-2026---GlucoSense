"""
Sensor ablation study for GlucoSense classification pipeline.

Trains the XGBoost classifier on different subsets of wearable sensor modalities,
keeping food-log features constant across all conditions. Reports balanced accuracy
and macro F1 for each subset.

Usage:
    python sensor_ablation.py

Data must be present at DESTINATION/ (or set DATA_ROOT in src/config.py).
"""

import sys
import pathlib
import importlib.util
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Patch config to point at local DESTINATION/ data directory
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("src.config", "src/config.py")
_cfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cfg)
_cfg.DATA_ROOT = pathlib.Path("DESTINATION")
sys.modules["src.config"] = _cfg

from src.train import build_dataset  # noqa: E402  (after config patch)

# ---------------------------------------------------------------------------
# Sensor subsets to evaluate
# Food-log features (carbs_3h, time_since_last_meal_min) are included in all
# conditions because they are not wearable sensor modalities.
# ---------------------------------------------------------------------------
FOOD_FEATURES = ["carbs_3h", "time_since_last_meal_min"]

SENSOR_SUBSETS = {
    "hr_ibi_only":          ["hr", "ibi"],
    "acc_only":             ["acc_mag"],
    "eda_only":             ["eda"],
    "temp_only":            ["temp"],
    "bvp_only":             ["bvp"],
    "hr_ibi_acc":           ["hr", "ibi", "acc_mag"],
    "hr_ibi_eda":           ["hr", "ibi", "eda"],
    "acc_eda":              ["acc_mag", "eda"],
    "all_wearable":         ["hr", "ibi", "eda", "temp", "bvp", "acc_mag"],
    "all_wearable_no_food": ["hr", "ibi", "eda", "temp", "bvp", "acc_mag"],  # no food
}

# The "all_wearable_no_food" condition strips food features so we can measure
# how much the food log contributes on top of sensors.
NO_FOOD_CONDITIONS = {"all_wearable_no_food"}


def run_cv(X: pd.DataFrame, y_enc: np.ndarray, groups: pd.Series, n_splits: int = 3):
    """Group K-Fold CV; returns (balanced_accuracy, macro_f1, per_class_f1_dict)."""
    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        num_class=len(np.unique(y_enc)),
    )

    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    oof = np.zeros(len(X), dtype=int)

    for tr, te in gkf.split(X, y_enc, groups):
        w_tr = compute_sample_weight(class_weight="balanced", y=y_enc[tr])
        model.fit(X.iloc[tr], y_enc[tr], sample_weight=w_tr)
        oof[te] = model.predict(X.iloc[te])

    bal_acc = balanced_accuracy_score(y_enc, oof)
    macro_f1 = f1_score(y_enc, oof, average="macro")
    per_class = f1_score(y_enc, oof, average=None)

    return bal_acc, macro_f1, per_class


def main():
    print("Loading dataset...")
    data = build_dataset()
    data = data.dropna(subset=["y"]).copy()

    groups = data["participant_id"]
    le = LabelEncoder()
    y_enc = le.fit_transform(data["y"])
    class_names = le.classes_

    print(f"Dataset: {data.shape[0]} rows, {groups.nunique()} participants")
    print(f"Classes: {list(class_names)}\n")

    results = []

    for condition, sensor_cols in SENSOR_SUBSETS.items():
        if condition in NO_FOOD_CONDITIONS:
            feature_cols = sensor_cols
        else:
            feature_cols = sensor_cols + FOOD_FEATURES

        # Replace inf (e.g. time_since_last_meal before first meal) with NaN → XGB handles it
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan)

        print(f"[{condition}]  features: {feature_cols}")
        bal_acc, macro_f1, per_class_f1 = run_cv(X, y_enc, groups)

        row = {
            "condition": condition,
            "features": feature_cols,
            "n_features": len(feature_cols),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
        }
        for cls, f1 in zip(class_names, per_class_f1):
            row[f"f1_{cls}"] = round(f1, 4)

        results.append(row)
        print(
            f"  balanced_acc={bal_acc:.4f}  macro_f1={macro_f1:.4f}  "
            + "  ".join(f"f1_{c}={f:.4f}" for c, f in zip(class_names, per_class_f1))
        )
        print()

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)

    print("=" * 80)
    print("ABLATION SUMMARY (sorted by macro F1)")
    print("=" * 80)
    cols = ["condition", "balanced_accuracy", "macro_f1"] + [f"f1_{c}" for c in class_names]
    print(df[cols].to_string(index=False))

    # Save results
    out_path = pathlib.Path("results/sensor_ablation.json")
    out_path.parent.mkdir(exist_ok=True)
    df.to_json(out_path, orient="records", indent=2)
    print(f"\nResults saved to {out_path}")

    return df, class_names


if __name__ == "__main__":
    main()
