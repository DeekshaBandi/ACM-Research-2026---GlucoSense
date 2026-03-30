"""
Sensor Ablation Study — GlucoSense
===================================
Trains a Random Forest regressor on different subsets of wearable sensor
modalities and compares predictive performance for blood glucose (mg/dL).

Modalities tested (individually and in combination):
  - HR / IBI  (heart rate + inter-beat interval)
  - ACC       (accelerometer: x, y, z + magnitude)
  - EDA       (electrodermal activity)
  - TEMP      (skin temperature)

Evaluation uses a temporal split (train on first 80% of readings by time,
test on last 20%) to avoid the temporal leakage in randomForestPractice.py.

All models also include baseline time + demographic features so comparisons
reflect the *added value* of each sensor subset.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

project_root     = Path(__file__).resolve().parent
destination_root = project_root / "DESTINATION"
RESAMPLE_WINDOW  = "5min"
RANDOM_STATE     = 42

# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DEXCOM (target)
# ──────────────────────────────────────────────────────────────────────────────

def load_dexcom(dest):
    ts_col  = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    ev_col  = "Event Type"
    glc_col = "Glucose Value (mg/dL)"

    frames = []
    for f in sorted(dest.glob("*/Dexcom_*.csv")):
        pid = int(f.stem.split("_")[-1])
        df  = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        if not {ts_col, ev_col, glc_col}.issubset(df.columns):
            continue
        df = df[df[ev_col] == "EGV"][[ts_col, glc_col]].copy()
        df = df.rename(columns={ts_col: "timestamp", glc_col: "glucose"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["glucose"]   = pd.to_numeric(df["glucose"], errors="coerce")
        df = df.dropna(subset=["timestamp", "glucose"])
        df["participant_id"] = pid
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. LOAD + RESAMPLE INDIVIDUAL SENSORS
# ──────────────────────────────────────────────────────────────────────────────

def resample_sensor(file_path, value_cols, window=RESAMPLE_WINDOW):
    """Read a sensor CSV, resample to window-level mean + std."""
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(
        df["datetime"].astype(str).str.strip(), errors="coerce"
    )
    df = df.dropna(subset=["datetime"]).set_index("datetime")

    agg = {}
    for col in value_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            agg[f"mean_{col}"] = s.resample(window).mean()
            agg[f"std_{col}"]  = s.resample(window).std()
    return pd.DataFrame(agg)


def load_participant_sensors(dest, pid):
    """Return dict of modality → resampled DataFrame for one participant."""
    p   = str(pid).zfill(3)
    base = dest / p
    sensors = {}

    # HR
    hr_file = base / f"HR_{p}.csv"
    if hr_file.exists():
        sensors["hr_ibi"] = resample_sensor(hr_file, ["hr"])

    # IBI — merge into hr_ibi group
    ibi_file = base / f"IBI_{p}.csv"
    if ibi_file.exists():
        ibi_df = resample_sensor(ibi_file, ["ibi"])
        if "hr_ibi" in sensors:
            sensors["hr_ibi"] = sensors["hr_ibi"].join(ibi_df, how="outer")
        else:
            sensors["hr_ibi"] = ibi_df

    # ACC — compute magnitude, then resample
    acc_file = base / f"ACC_{p}.csv"
    if acc_file.exists():
        df = pd.read_csv(acc_file)
        df.columns = df.columns.str.strip()
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"]).set_index("datetime")
        for c in ["acc_x", "acc_y", "acc_z"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["acc_mag"] = np.sqrt(df["acc_x"]**2 + df["acc_y"]**2 + df["acc_z"]**2)
        agg = {}
        for col in ["acc_x", "acc_y", "acc_z", "acc_mag"]:
            agg[f"mean_{col}"] = df[col].resample(RESAMPLE_WINDOW).mean()
            agg[f"std_{col}"]  = df[col].resample(RESAMPLE_WINDOW).std()
        sensors["acc"] = pd.DataFrame(agg)

    # EDA
    eda_file = base / f"EDA_{p}.csv"
    if eda_file.exists():
        sensors["eda"] = resample_sensor(eda_file, ["eda"])

    # TEMP
    temp_file = base / f"TEMP_{p}.csv"
    if temp_file.exists():
        sensors["temp"] = resample_sensor(temp_file, ["temp"])

    return sensors


# ──────────────────────────────────────────────────────────────────────────────
# 3. BUILD MERGED DATASET
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(dest):
    dexcom = load_dexcom(dest)

    # Demographics
    demo_path = dest / "Demographics.csv"
    if demo_path.exists():
        demo = pd.read_csv(demo_path)
        demo.columns = demo.columns.str.strip()
        demo = demo.rename(columns={"ID": "participant_id"})
        demo["participant_id"] = pd.to_numeric(demo["participant_id"], errors="coerce").astype("Int64")
        demo["HbA1c"]  = pd.to_numeric(demo["HbA1c"], errors="coerce")
        demo["Gender"] = demo["Gender"].astype(str).str.strip().str.upper()
        demo = demo[["participant_id", "Gender", "HbA1c"]]
        dexcom["participant_id"] = dexcom["participant_id"].astype("Int64")
        dexcom = dexcom.merge(demo, on="participant_id", how="left")

    # Time features
    dexcom["hour"]       = dexcom["timestamp"].dt.hour
    dexcom["minute"]     = dexcom["timestamp"].dt.minute
    dexcom["day_of_week"]= dexcom["timestamp"].dt.dayofweek

    # Floor timestamp to 5-min for sensor alignment
    dexcom["ts_5min"] = dexcom["timestamp"].dt.floor(RESAMPLE_WINDOW)

    parts = []
    for pid in sorted(dexcom["participant_id"].dropna().unique()):
        p_df    = dexcom[dexcom["participant_id"] == pid].copy()
        sensors = load_participant_sensors(dest, int(pid))

        for modality, s_df in sensors.items():
            s_df = s_df.copy()
            s_df.index = s_df.index.floor(RESAMPLE_WINDOW)
            s_df.index.name = "ts_5min"
            s_df = s_df.reset_index().drop_duplicates(subset=["ts_5min"])
            p_df = p_df.merge(s_df, on="ts_5min", how="left")

        parts.append(p_df)

    return pd.concat(parts, ignore_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# 4. ABLATION DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

BASELINE_COLS = ["hour", "minute", "day_of_week", "participant_id", "HbA1c", "Gender"]

SENSOR_COLS = {
    "hr_ibi": ["mean_hr", "std_hr", "mean_ibi", "std_ibi"],
    "acc":    ["mean_acc_x", "std_acc_x", "mean_acc_y", "std_acc_y",
               "mean_acc_z", "std_acc_z", "mean_acc_mag", "std_acc_mag"],
    "eda":    ["mean_eda", "std_eda"],
    "temp":   ["mean_temp", "std_temp"],
}

ABLATION_CONFIGS = {
    "Baseline (time + demographics, no wearables)": [],
    "HR / IBI only":                                ["hr_ibi"],
    "Accelerometer only":                           ["acc"],
    "EDA only":                                     ["eda"],
    "Temperature only":                             ["temp"],
    "HR + Accelerometer":                           ["hr_ibi", "acc"],
    "HR + EDA":                                     ["hr_ibi", "eda"],
    "Accelerometer + EDA":                          ["acc", "eda"],
    "All wearable modalities":                      ["hr_ibi", "acc", "eda", "temp"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN + EVALUATE (temporal split)
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(df):
    # Temporal split — sort by time, take last 20% as test
    df = df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    train_df  = df.iloc[:split_idx]
    test_df   = df.iloc[split_idx:]

    records = []

    for config_name, modality_keys in ABLATION_CONFIGS.items():
        # Build feature list from baseline + requested sensor groups
        feature_cols = BASELINE_COLS.copy()
        for key in modality_keys:
            feature_cols += [c for c in SENSOR_COLS[key] if c in df.columns]
        feature_cols = [c for c in feature_cols if c in df.columns]

        X_train = train_df[feature_cols].copy()
        X_test  = test_df[feature_cols].copy()
        y_train = train_df["glucose"]
        y_test  = test_df["glucose"]

        # Drop rows where any feature or target is NaN
        tr_mask = X_train.notna().all(axis=1) & y_train.notna()
        te_mask = X_test.notna().all(axis=1)  & y_test.notna()
        X_train, y_train = X_train[tr_mask], y_train[tr_mask]
        X_test,  y_test  = X_test[te_mask],  y_test[te_mask]

        cat_cols = X_train.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()

        transformers = []
        if cat_cols:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
        if num_cols:
            transformers.append(("num", "passthrough", num_cols))

        model = Pipeline([
            ("preprocess", ColumnTransformer(transformers)),
            ("regressor",  RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ])

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2   = r2_score(y_test, preds)

        records.append({
            "Configuration":  config_name,
            "MAE (mg/dL)":    round(mae, 3),
            "RMSE (mg/dL)":   round(rmse, 3),
            "R²":             round(r2, 4),
            "# Features":     len(feature_cols),
            "Test rows":      len(y_test),
        })

        print(f"  {config_name:<48}  MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}  (n={len(y_test)})")

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building merged dataset (this may take a moment)...")
    df = build_dataset(destination_root)
    print(f"Total rows after merge: {len(df):,}\n")

    print("Running sensor ablation (temporal split: 80% train / 20% test by time)\n")
    results = run_ablation(df)

    print("\n" + "=" * 75)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 75)
    print(results.to_string(index=False))

    out = project_root / "results" / "sensor_ablation_results.csv"
    out.parent.mkdir(exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nResults saved to {out}")
