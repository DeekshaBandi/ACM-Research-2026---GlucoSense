"""
Random Forest Regression Boilerplate Template

How to use:
1. Replace 'your_file.csv' with your dataset path
2. Replace 'target_column_name' with your actual target column
3. Run: python random_forest_template.py
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from pathlib import Path

HYPOGLYCEMIA_THRESHOLD = 70
HYPERGLYCEMIA_THRESHOLD = 180

# --------------------------
# 1. LOAD DATA
# --------------------------

# Use all 16 participant Dexcom files and merge participant demographics.
project_root = Path(__file__).resolve().parent
destination_root = project_root / "DESTINATION"

dexcom_files = sorted(destination_root.glob("*/Dexcom_*.csv"))
if len(dexcom_files) == 0:
    raise FileNotFoundError("No Dexcom files found under DESTINATION/*/Dexcom_*.csv")

timestamp_col = "Timestamp (YYYY-MM-DDThh:mm:ss)"
event_type_col = "Event Type"
glucose_col = "Glucose Value (mg/dL)"

dexcom_frames = []
for file_path in dexcom_files:
    participant_id = int(file_path.stem.split("_")[-1])
    raw_df = pd.read_csv(file_path)
    raw_df.columns = raw_df.columns.str.strip()

    if not {timestamp_col, event_type_col, glucose_col}.issubset(raw_df.columns):
        continue

    participant_df = raw_df[[timestamp_col, event_type_col, glucose_col]].copy()
    participant_df = participant_df[participant_df[event_type_col] == "EGV"]
    participant_df = participant_df.rename(
        columns={
            timestamp_col: "timestamp",
            glucose_col: "glucose",
        }
    )
    participant_df["participant_id"] = participant_id
    participant_df["timestamp"] = pd.to_datetime(participant_df["timestamp"], errors="coerce")
    participant_df["glucose"] = pd.to_numeric(participant_df["glucose"], errors="coerce")
    participant_df = participant_df.dropna(subset=["timestamp", "glucose"])
    dexcom_frames.append(participant_df)

if len(dexcom_frames) == 0:
    raise ValueError("No valid EGV glucose rows found in Dexcom files.")

df = pd.concat(dexcom_frames, ignore_index=True)

# Time-derived features avoid high-cardinality raw timestamp encoding.
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["day_of_week"] = df["timestamp"].dt.dayofweek

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

# Keep only modeling columns.
df = df[["glucose", "participant_id", "hour", "minute", "day_of_week", "Gender", "HbA1c"]]

# Clinical factors for distribution reporting (not used as model features).
df["hypoglycemia_factor"] = (df["glucose"] < HYPOGLYCEMIA_THRESHOLD).astype(int)
df["hyperglycemia_factor"] = (df["glucose"] > HYPERGLYCEMIA_THRESHOLD).astype(int)

print("Dataset shape:", df.shape)
print("\nDataset dtypes:\n", df.dtypes)
print(df.head())
print(
    "\nHypoglycemia (<70 mg/dL):",
    f"{df['hypoglycemia_factor'].sum()}",
    f"({df['hypoglycemia_factor'].mean() * 100:.2f}%)",
)
print(
    "Hyperglycemia (>180 mg/dL):",
    f"{df['hyperglycemia_factor'].sum()}",
    f"({df['hyperglycemia_factor'].mean() * 100:.2f}%)",
)


# --------------------------
# 2. DEFINE TARGET + FEATURES
# --------------------------

target_column = "glucose"

X = df.drop(columns=[target_column, "hypoglycemia_factor", "hyperglycemia_factor"])
y = df[target_column]


# --------------------------
# 3. HANDLE CATEGORICAL DATA (if any)
# --------------------------

# Automatically detect categorical columns
categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns
numeric_cols = X.select_dtypes(include=["number"]).columns

print("Categorical columns:", list(categorical_cols))
print("Numeric columns:", list(numeric_cols))

# One-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)


# --------------------------
# 4. TRAIN TEST SPLIT
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# --------------------------
# 5. CREATE RANDOM FOREST MODEL
# --------------------------

rf_model = RandomForestRegressor(
    n_estimators=300,      # number of trees
    max_depth=None,        # trees grow fully
    random_state=42,
    n_jobs=-1              # use all CPU cores
)

# Pipeline = preprocessing + model in one object
model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("regressor", rf_model)
])


# --------------------------
# 6. TRAIN MODEL
# --------------------------

model.fit(X_train, y_train)


# --------------------------
# 7. MAKE PREDICTIONS
# --------------------------

predictions = model.predict(X_test)


# --------------------------
# 8. EVALUATE MODEL
# --------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nModel Performance:")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2 Score:", round(r2, 4))


# --------------------------
# 9. FEATURE IMPORTANCE
# --------------------------

# Extract feature names after encoding
ohe = model.named_steps["preprocess"].named_transformers_["cat"]

encoded_cat_features = []
if len(categorical_cols) > 0:
    encoded_cat_features = ohe.get_feature_names_out(categorical_cols)

all_feature_names = list(encoded_cat_features) + list(numeric_cols)

importances = model.named_steps["regressor"].feature_importances_

print("\nTop Feature Importances:")
sorted_idx = np.argsort(importances)[::-1]

for i in sorted_idx[:10]:
    print(all_feature_names[i], ":", round(importances[i], 4))
