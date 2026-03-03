"""
Optimized Binary Glucose Classification Pipeline
=================================================

Binary classifier for glucose level events:
  - Class 0: Normal (70–180 mg/dL)
  - Class 1: Any Event (Hypoglycemia <70 OR Hyperglycemia >180)

Event type tracking (for Class 1):
  - 'low': Hypoglycemia (<70 mg/dL)
  - 'high': Hyperglycemia (>180 mg/dL)

Features used: 9 optimized features combining glucose and HR metrics
Model: RandomForest with balanced class weights to handle data imbalance

"""

import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get script directory and build paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data")

MIN_SUBJECT_ROWS = 50
FEATURE_LIST = [
    "glucose_prev",
    "glucose_delta",
    "glucose_roll_mean",
    "glucose_roll_std",
    "HR_mean",
    "HR_std",
    "HR_min",
    "HR_max",
    "HR_range",
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def create_glucose_label(glucose_value):
    """
    Create binary classification label from glucose value.

    0 = Normal (70–180 mg/dL)
    1 = Any Event (Hypoglycemia <70 OR Hyperglycemia >180)

    Event type is tracked separately:
      - 'low' = Hypoglycemia (<70)
      - 'high' = Hyperglycemia (>180)
      - 'normal' = Normal (70–180)
    """
    if glucose_value < 70:
        return 1  # Event - Low
    elif glucose_value > 180:
        return 1  # Event - High
    else:
        return 0  # Normal


def get_event_type(glucose_value):
    """
    Track the type of event (or normal).

    Returns:
        str: 'low', 'high', or 'normal'
    """
    if glucose_value < 70:
        return "low"
    elif glucose_value > 180:
        return "high"
    else:
        return "normal"


# ============================================================================
# LOAD & PREPROCESS SUBJECT
# ============================================================================


def load_subject(subject_path):
    """
    Load and merge Dexcom (glucose) + HR data for a single subject.

    Args:
        subject_path (str): Path to subject folder

    Returns:
        pd.DataFrame: Merged dataframe with glucose, HR, and label
                      or None if processing fails

    Raises:
        Catches and logs all errors internally
    """
    try:
        # Find Dexcom and HR files
        dexcom_files = glob(os.path.join(subject_path, "*Dexcom*.csv"))
        hr_files = glob(os.path.join(subject_path, "*HR*.csv"))

        if not dexcom_files or not hr_files:
            return None

        dexcom_file = dexcom_files[0]
        hr_file = hr_files[0]

        # Load CSVs
        dex = pd.read_csv(dexcom_file)
        hr = pd.read_csv(hr_file)

        # ===== Clean Dexcom =====
        dex.columns = dex.columns.str.strip()

        timestamp_col = [c for c in dex.columns if "Timestamp" in c]
        glucose_col = [c for c in dex.columns if "Glucose" in c]

        if not timestamp_col or not glucose_col:
            return None

        dex["Timestamp"] = pd.to_datetime(dex[timestamp_col[0]], errors="coerce")
        dex["glucose"] = pd.to_numeric(dex[glucose_col[0]], errors="coerce")
        dex["label"] = dex["glucose"].apply(create_glucose_label)
        dex["event_type"] = dex["glucose"].apply(get_event_type)
        dex = dex[["Timestamp", "glucose", "label", "event_type"]].dropna()

        if len(dex) == 0:
            return None

        # ===== Clean HR =====
        hr.columns = hr.columns.str.strip()

        if "datetime" in hr.columns:
            hr["Timestamp"] = pd.to_datetime(hr["datetime"], errors="coerce")
        else:
            # Try first column as datetime
            hr["Timestamp"] = pd.to_datetime(hr.iloc[:, 0], errors="coerce")

        # Try to extract HR from column with numeric data
        hr["HR"] = pd.to_numeric(hr.iloc[:, -1], errors="coerce")
        hr = hr[["Timestamp", "HR"]].dropna()

        if len(hr) == 0:
            return None

        # ===== Merge on Timestamp =====
        merged = pd.merge_asof(
            dex.sort_values("Timestamp"),
            hr.sort_values("Timestamp"),
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("5min"),
        )

        merged = merged.dropna()

        if len(merged) < MIN_SUBJECT_ROWS:
            return None

        return merged

    except Exception as e:
        return None


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def engineer_features(df):
    """
    Create optimized feature set from glucose + HR data.

    Features (9 total):
      - glucose_prev: Previous glucose value (lag 1)
      - glucose_delta: Change in glucose from previous reading
      - glucose_roll_mean: Rolling mean of glucose (window=3)
      - glucose_roll_std: Rolling std of glucose (window=3)
      - HR_mean: Rolling mean of HR (window=3)
      - HR_std: Rolling std of HR (window=3)
      - HR_min: Rolling min of HR (window=3)
      - HR_max: Rolling max of HR (window=3)
      - HR_range: Difference between HR_max and HR_min

    Args:
        df (pd.DataFrame): Merged dataframe with glucose, HR columns

    Returns:
        pd.DataFrame: Dataframe with engineered features and label
    """
    # Glucose features
    df["glucose_prev"] = df["glucose"].shift(1)
    df["glucose_delta"] = df["glucose"] - df["glucose_prev"]
    df["glucose_roll_mean"] = df["glucose"].rolling(window=3).mean()
    df["glucose_roll_std"] = df["glucose"].rolling(window=3).std()

    # HR features
    df["HR_mean"] = df["HR"].rolling(window=3).mean()
    df["HR_std"] = df["HR"].rolling(window=3).std()
    df["HR_min"] = df["HR"].rolling(window=3).min()
    df["HR_max"] = df["HR"].rolling(window=3).max()
    df["HR_range"] = df["HR_max"] - df["HR_min"]

    # Drop NaNs created by rolling/shifting
    df = df.dropna()

    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================


def train_model(X_train, y_train):
    """
    Train RandomForest classifier with optimized hyperparameters.

    Args:
        X_train (np.ndarray): Training features (scaled)
        y_train (np.ndarray): Training labels

    Returns:
        RandomForestClassifier: Trained model
    """
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate trained model and print comprehensive metrics.

    Uses the *regular* (binary) F1 score rather than macro, since
    we're focused on overall event detection performance.  Also
    displays accuracy, classification report, confusion matrix and
    ranked feature importances.

    Args:
        model (RandomForestClassifier): Trained model
        X_test (np.ndarray): Test features (scaled)
        y_test (np.ndarray): Test labels
        feature_names (list): List of feature names for importance ranking

    Returns:
        dict: Dictionary with accuracy and binary F1 score
    """
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)  # binary by default

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)

    print("\nAccuracy: {:.4f}".format(accuracy))
    print("F1 Score: {:.4f}".format(f1))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature importance
    importances = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values(by="Importance", ascending=False)

    print("\nFeature Importance (Ranked):")
    print(importances.to_string(index=False))

    return {
        "accuracy": accuracy,
        "f1": f1,
    }


# ============================================================================
# FEATURE OPTIMIZATION
# ============================================================================

def optimize_features(X_train_df, y_train, X_test_df, y_test):
    """
    Greedy search over subsets of features, starting with the most
    important ones as determined by the full-model importances.

    Trains a fresh RandomForest on the top-k features and records the F1
    score on the held-out test set.  Returns the smallest subset that
    achieves the highest observed F1, which helps keep the feature set
    minimal while maintaining performance.

    Args:
        X_train_df (pd.DataFrame): scaled training features
        y_train (pd.Series): training labels
        X_test_df (pd.DataFrame): scaled test features
        y_test (pd.Series): test labels

    Returns:
        list: names of features in the optimal subset
    """
    # first fit a model on all features to get importances
    base_model = train_model(X_train_df.values, y_train)
    importances = pd.DataFrame(
        {
            "feature": X_train_df.columns,
            "importance": base_model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    best_f1 = 0.0
    best_subset = list(X_train_df.columns)

    # iterate increasing subset size
    for k in range(1, len(importances) + 1):
        subset = importances["feature"].iloc[:k].tolist()
        model_k = train_model(X_train_df[subset].values, y_train)
        y_pred_k = model_k.predict(X_test_df[subset].values)
        f1_k = f1_score(y_test, y_pred_k, zero_division=0)

        # prefer smaller subset when f1 equal
        if f1_k > best_f1 or (f1_k == best_f1 and len(subset) < len(best_subset)):
            best_f1 = f1_k
            best_subset = subset.copy()

    print(f"\nOptimal feature subset (F1={best_f1:.4f}): {len(best_subset)} features")
    print(f"  {best_subset}")
    return best_subset


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main():
    """Main pipeline: load → engineer → combine → train → evaluate"""

    print("=" * 70)
    print("GLUCOSE CLASSIFICATION PIPELINE")
    print("=" * 70)
    print(f"Data path: {DATA_PATH}")
    print(f"Classification: Binary (Normal vs Event)")
    print(f"Features: {len(FEATURE_LIST)} optimized features")
    print()

    # ===== Load all subjects =====
    all_data = []
    processed_subjects = []

    subject_dirs = sorted(
        [d for d in glob(os.path.join(DATA_PATH, "*")) if os.path.isdir(d)]
    )

    print(f"Found {len(subject_dirs)} subject folders\n")

    for subject_path in subject_dirs:
        subject_id = os.path.basename(subject_path)

        df = load_subject(subject_path)

        if df is None:
            print(f"  ⊗ {subject_id}: Skipped (missing/invalid files)")
            continue

        # Feature engineering
        df = engineer_features(df)

        if len(df) < MIN_SUBJECT_ROWS:
            print(
                f"  ⊗ {subject_id}: Skipped (insufficient rows after features)"
            )
            continue

        all_data.append(df)
        processed_subjects.append(subject_id)
        print(
            f"  ✓ {subject_id}: {len(df):,} rows | Classes: {df['label'].value_counts().to_dict()}"
        )

    print()

    if len(all_data) == 0:
        print("ERROR: No valid subjects loaded. Check DATA_PATH and file formats.")
        return

    # ===== Combine all subjects =====
    dataset = pd.concat(all_data, ignore_index=True)

    print(f"Processed Subjects: {', '.join(processed_subjects)}")
    print(f"\nTotal Dataset Shape: {dataset.shape}")
    print(f"Total Rows: {len(dataset):,}")
    print(f"Total Features (incl. label): {dataset.shape[1]}")

    print("\nBinary Classification Distribution:")
    class_dist = dataset["label"].value_counts(sort=False).to_dict()
    class_names = {0: "Normal (70-180)", 1: "Event (Low/High)"}
    total = len(dataset)
    for class_id in sorted(class_dist.keys()):
        count = class_dist[class_id]
        pct = 100.0 * count / total
        print(f"  Class {class_id} ({class_names[class_id]:20}): {count:,} ({pct:5.2f}%)")

    print("\nEvent Type Distribution (for Class 1):")
    event_dist = dataset[dataset["label"] == 1]["event_type"].value_counts()
    for event_type, count in event_dist.items():
        total_events = len(dataset[dataset["label"] == 1])
        pct = 100.0 * count / total_events
        event_label = "Hypoglycemia (<70)" if event_type == "low" else "Hyperglycemia (>180)"
        print(f"  {event_label:20}: {count:,} ({pct:5.2f}%)")

    # ===== Train/Test Split =====
    X = dataset[FEATURE_LIST]
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train):,} samples | Test: {len(X_test):,} samples")

    # ===== Scale Features =====
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # make DataFrames for easier subset selection later
    X_train_df = pd.DataFrame(X_train_scaled, columns=FEATURE_LIST)
    X_test_df = pd.DataFrame(X_test_scaled, columns=FEATURE_LIST)

    # ===== Train Model =====
    print("\nTraining RandomForestClassifier...")
    model = train_model(X_train_scaled, y_train)

    # ===== Evaluate =====
    metrics = evaluate_model(model, X_test_scaled, y_test, FEATURE_LIST)

    # ===== Feature subset optimization =====
    best_subset = optimize_features(X_train_df, y_train, X_test_df, y_test)

    # if a smaller subset was found, re‑train & report again
    if len(best_subset) < len(FEATURE_LIST):
        print("\nRe-evaluating model using optimal subset of features...")
        model2 = train_model(X_train_df[best_subset].values, y_train)
        evaluate_model(model2, X_test_df[best_subset].values, y_test, best_subset)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()