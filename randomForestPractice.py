"""
Glucose Excursion Detection (rare-event focused).

This script:
1) Loads all 16 Dexcom participant files
2) Builds low-leakage temporal + demographic features only
3) Defines excursion target: glucose < 70 OR glucose > 180
4) Evaluates models with LeaveOneGroupOut (group = participant_id)
5) Prioritizes rare-event detection metrics over raw accuracy
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

HYPOGLYCEMIA_THRESHOLD = 70
HYPERGLYCEMIA_THRESHOLD = 180
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    """Load Dexcom rows and merge demographics across all participants."""
    project_root = Path(__file__).resolve().parent
    destination_root = project_root / "DESTINATION"
    dexcom_files = sorted(destination_root.glob("*/Dexcom_*.csv"))
    if not dexcom_files:
        raise FileNotFoundError("No Dexcom files found under DESTINATION/*/Dexcom_*.csv")

    timestamp_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    event_type_col = "Event Type"
    glucose_col = "Glucose Value (mg/dL)"

    frames: list[pd.DataFrame] = []
    for file_path in dexcom_files:
        participant_id = int(file_path.stem.split("_")[-1])
        raw_df = pd.read_csv(file_path)
        raw_df.columns = raw_df.columns.str.strip()

        required_cols = {timestamp_col, event_type_col, glucose_col}
        if not required_cols.issubset(raw_df.columns):
            continue

        participant_df = raw_df[[timestamp_col, event_type_col, glucose_col]].copy()
        participant_df = participant_df[participant_df[event_type_col] == "EGV"]
        participant_df = participant_df.rename(
            columns={timestamp_col: "timestamp", glucose_col: "glucose"}
        )
        participant_df["participant_id"] = participant_id
        participant_df["timestamp"] = pd.to_datetime(participant_df["timestamp"], errors="coerce")
        participant_df["glucose"] = pd.to_numeric(participant_df["glucose"], errors="coerce")
        participant_df = participant_df.dropna(subset=["timestamp", "glucose"])
        frames.append(participant_df)

    if not frames:
        raise ValueError("No valid EGV glucose rows found in Dexcom files.")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["participant_id", "timestamp"]).reset_index(drop=True)

    # Temporal/context features.
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Glucose-history features for stronger excursion detectability.
    for lag in [1, 2, 3, 6, 12]:
        df[f"glucose_lag_{lag}"] = df.groupby("participant_id")["glucose"].shift(lag)
    for window in [3, 6, 12]:
        df[f"glucose_roll_mean_{window}"] = df.groupby("participant_id")["glucose"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f"glucose_roll_std_{window}"] = df.groupby("participant_id")["glucose"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=2).std()
        )
    df["glucose_delta_1"] = df["glucose_lag_1"] - df["glucose_lag_2"]
    df["glucose_delta_3"] = df["glucose_lag_1"] - df["glucose_lag_3"]

    demographics_path = destination_root / "Demographics.csv"
    if demographics_path.exists():
        demo_df = pd.read_csv(demographics_path)
        demo_df.columns = demo_df.columns.str.strip()
        demo_df["ID"] = pd.to_numeric(demo_df["ID"], errors="coerce")
        demo_df = demo_df.rename(columns={"ID": "participant_id"})
        demo_df["participant_id"] = demo_df["participant_id"].astype("Int64")
        demo_df["HbA1c"] = pd.to_numeric(demo_df["HbA1c"], errors="coerce")
        demo_df["Gender"] = demo_df["Gender"].astype(str).str.strip().str.upper()
        demo_df = demo_df[["participant_id", "Gender", "HbA1c"]]
        df["participant_id"] = df["participant_id"].astype("Int64")
        df = df.merge(demo_df, on="participant_id", how="left")

    # Fill demographics missingness with robust defaults.
    df["Gender"] = df["Gender"].fillna("UNKNOWN")
    df["HbA1c"] = df["HbA1c"].fillna(df["HbA1c"].median())
    return df.dropna().reset_index(drop=True)


def evaluate_with_logo(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model: Pipeline,
    use_balanced_sample_weight: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold probabilities with LeaveOneGroupOut."""
    logo = LeaveOneGroupOut()
    y_true_all: list[np.ndarray] = []
    y_proba_all: list[np.ndarray] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = clone(model)
        if use_balanced_sample_weight:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
            fold_model.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            fold_model.fit(X_train, y_train)

        y_proba = fold_model.predict_proba(X_test)[:, 1]

        y_true_all.append(y_test.to_numpy())
        y_proba_all.append(y_proba)

    return np.concatenate(y_true_all), np.concatenate(y_proba_all)


def find_best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Pick threshold that maximizes F1 on out-of-fold probabilities."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_values = (2 * precision * recall) / (precision + recall + 1e-12)
    # thresholds has one fewer item than precision/recall.
    if len(thresholds) == 0:
        return 0.5
    best_idx = int(np.nanargmax(f1_values[:-1]))
    return float(thresholds[best_idx])


def get_feature_importance(model: Pipeline) -> pd.Series:
    """Extract importance-like signal for linear/tree models."""
    preprocessor = model.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_[0])
    else:
        importances = np.zeros(len(feature_names), dtype=float)

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def main() -> None:
    df = load_dataset()
    df["excursion"] = (
        (df["glucose"] < HYPOGLYCEMIA_THRESHOLD) | (df["glucose"] > HYPERGLYCEMIA_THRESHOLD)
    ).astype(int)

    # Use temporal, demographic, and glucose-history predictors.
    feature_cols = [
        "hour",
        "minute",
        "day_of_week",
        "hour_sin",
        "hour_cos",
        "Gender",
        "HbA1c",
        "glucose_lag_1",
        "glucose_lag_2",
        "glucose_lag_3",
        "glucose_lag_6",
        "glucose_lag_12",
        "glucose_roll_mean_3",
        "glucose_roll_mean_6",
        "glucose_roll_mean_12",
        "glucose_roll_std_3",
        "glucose_roll_std_6",
        "glucose_roll_std_12",
        "glucose_delta_1",
        "glucose_delta_3",
    ]
    X = df[feature_cols].copy()
    y = df["excursion"].copy()
    groups = df["participant_id"].copy()

    categorical_cols = ["Gender"]
    numeric_cols = [col for col in feature_cols if col != "Gender"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    models = {
        "Logistic Regression (balanced)": (
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
            False,
        ),
        "Random Forest (balanced)": (
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1,
                        ),
                    ),
                ]
            ),
            False,
        ),
        "Gradient Boosting (balanced sample weights)": (
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
                ]
            ),
            True,
        ),
    }

    print(f"Dataset shape: {df.shape}")
    print(f"Excursion positives: {int(y.sum())} ({y.mean() * 100:.2f}%)")
    print(f"Excursion negatives: {int((1 - y).sum())} ({(1 - y.mean()) * 100:.2f}%)")
    print(f"Participants (LOGO folds): {df['participant_id'].nunique()}")

    plt.figure(figsize=(8, 6))
    model_results = {}

    for model_name, (pipeline, use_balanced_sample_weight) in models.items():
        y_true, y_proba = evaluate_with_logo(
            X, y, groups, pipeline, use_balanced_sample_weight
        )
        best_threshold = find_best_f1_threshold(y_true, y_proba)
        y_pred = (y_proba >= best_threshold).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        r2 = r2_score(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

        fitted_for_importance = clone(pipeline)
        if use_balanced_sample_weight:
            sample_weight_all = compute_sample_weight(class_weight="balanced", y=y)
            fitted_for_importance.fit(X, y, model__sample_weight=sample_weight_all)
        else:
            fitted_for_importance.fit(X, y)
        importance = get_feature_importance(fitted_for_importance)

        model_results[model_name] = {
            "precision": precision,
            "recall_excursion": recall,
            "f1": f1,
            "r2": r2,
            "roc_auc": roc_auc,
            "best_threshold": best_threshold,
            "confusion_matrix": cm,
            "importance": importance,
            "report": classification_report(y_true, y_pred, zero_division=0),
        }

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Excursion Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    roc_path = output_dir / "excursion_roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()

    print("\n=== Rare Event Detection Results (Excursion Class = 1) ===")
    for model_name, result in model_results.items():
        print(f"\n{model_name}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall (excursion class): {result['recall_excursion']:.4f}")
        print(f"F1-score: {result['f1']:.4f}")
        print(f"R2-score: {result['r2']:.4f}")
        print(f"ROC-AUC: {result['roc_auc']:.4f}")
        print(f"Best threshold (for F1): {result['best_threshold']:.4f}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:")
        print(result["confusion_matrix"])
        print("Top feature importance:")
        for feature_name, feature_value in result["importance"].head(8).items():
            print(f"  {feature_name}: {feature_value:.4f}")

    print(f"\nROC curve saved to: {roc_path}")


if __name__ == "__main__":
    main()
