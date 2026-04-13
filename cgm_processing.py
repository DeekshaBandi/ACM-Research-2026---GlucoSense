"""
CGM Processing Pipeline
-----------------------
For each participant-day, computes:
  - mean, SD, CV
  - TIR (70–180), TAR (>180), TBR (<70)
  - MAGE (excursions > 1 SD, using peaks/troughs)
  - n_readings (quality filter)

Then assigns binary label: high vs low glycemic variability using a
cohort-relative threshold on CV.

Labeling strategy
-----------------
The clinical CV threshold of 36% (Danne et al., 2017) is designed for
T1D/T2D populations. This dataset contains normoglycemic participants
(mean CV ~16.7%), so a cohort-relative split is used instead:

  LABEL_STRATEGY = "median"   → days above cohort median CV = "high"
  LABEL_STRATEGY = "p75"      → top quartile of CV days = "high"

The median split gives a balanced ~50/50 label; the p75 split gives a
25/75 split emphasising the most variable days. Either is defensible
as a feasibility study in a non-diabetic cohort. We use "median" as
the primary label and report p75 as a sensitivity column.

Output: data/cgm_daily_labels.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "cgm_daily_labels.csv")

# Minimum readings per day to include (5-min intervals → 288 max; require ≥50%)
MIN_READINGS = 144


def load_cgm(participant_id: str) -> pd.DataFrame:
    """Load and clean EGV rows from a Dexcom CSV."""
    path = os.path.join(DATA_DIR, participant_id, f"Dexcom_{participant_id}.csv")
    df = pd.read_csv(path)
    df = df[df["Event Type"] == "EGV"].copy()
    df["timestamp"] = pd.to_datetime(df["Timestamp (YYYY-MM-DDThh:mm:ss)"])
    df["glucose"] = pd.to_numeric(df["Glucose Value (mg/dL)"], errors="coerce")
    df = df.dropna(subset=["glucose"])
    # Drop Dexcom "Low" / "High" sentinel strings that survived to_numeric as NaN — already dropped
    # Also drop physiologically implausible values
    df = df[(df["glucose"] >= 40) & (df["glucose"] <= 400)]
    df["date"] = df["timestamp"].dt.date
    df["participant"] = participant_id
    return df[["participant", "timestamp", "date", "glucose"]].sort_values("timestamp")


def compute_mage(glucose: np.ndarray) -> float:
    """
    MAGE: mean of glucose excursions whose amplitude exceeds 1 SD.

    Algorithm (Service et al. 1970 / Kovatchev refinement):
      1. Find all local peaks and troughs (order=3 window).
      2. For consecutive peak→trough or trough→peak pairs, compute amplitude.
      3. Keep excursions where amplitude > 1 SD of the day's glucose.
      4. Return mean of kept amplitudes (NaN if none qualify).
    """
    if len(glucose) < 7:
        return np.nan

    sd = np.std(glucose, ddof=1)
    if sd == 0:
        return 0.0

    peaks = argrelextrema(glucose, np.greater_equal, order=3)[0]
    troughs = argrelextrema(glucose, np.less_equal, order=3)[0]

    # Merge and sort turning points with labels
    turning_points = sorted(
        [(i, "peak") for i in peaks] + [(i, "trough") for i in troughs],
        key=lambda x: x[0],
    )

    excursions = []
    for i in range(len(turning_points) - 1):
        idx_a, type_a = turning_points[i]
        idx_b, type_b = turning_points[i + 1]
        if type_a != type_b:  # alternating peak/trough
            amplitude = abs(glucose[idx_a] - glucose[idx_b])
            if amplitude > sd:
                excursions.append(amplitude)

    return float(np.mean(excursions)) if excursions else np.nan


def compute_daily_metrics(group: pd.DataFrame) -> dict:
    """Compute all CGM metrics for a single participant-day."""
    g = group["glucose"].values
    n = len(g)

    mean_g = np.mean(g)
    sd_g = np.std(g, ddof=1) if n > 1 else np.nan
    cv = (sd_g / mean_g * 100) if mean_g > 0 else np.nan

    tir = np.mean((g >= 70) & (g <= 180)) * 100
    tar = np.mean(g > 180) * 100
    tbr = np.mean(g < 70) * 100

    mage = compute_mage(g)

    return {
        "n_readings": n,
        "mean_glucose": round(mean_g, 2),
        "sd_glucose": round(sd_g, 2) if not np.isnan(sd_g) else np.nan,
        "cv": round(cv, 2) if not np.isnan(cv) else np.nan,
        "tir_70_180": round(tir, 2),
        "tar_180": round(tar, 2),
        "tbr_70": round(tbr, 2),
        "mage": round(mage, 2) if not np.isnan(mage) else np.nan,
    }


def process_all() -> tuple[pd.DataFrame, dict]:
    participants = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
    )
    print(f"Found {len(participants)} participants: {participants}")

    records = []
    for pid in participants:
        try:
            df = load_cgm(pid)
        except FileNotFoundError:
            print(f"  [{pid}] Dexcom file not found — skipping")
            continue

        for date, group in df.groupby("date"):
            metrics = compute_daily_metrics(group)
            metrics["participant"] = pid
            metrics["date"] = date
            records.append(metrics)

    result = pd.DataFrame(records)

    # Quality filter
    before = len(result)
    result = result[result["n_readings"] >= MIN_READINGS].copy()
    after = len(result)
    print(f"Dropped {before - after} days with <{MIN_READINGS} readings "
          f"({after} participant-days retained)")

    # ── Binary labels ────────────────────────────────────────────────────────
    # DESIGN DECISION: Labels are cohort-relative (median / p75 splits), NOT
    # based on a clinical CV threshold (e.g., 36%).  This classifies relatively
    # higher vs lower daily glycemic variability *within this cohort*.  The
    # manuscript must frame findings accordingly — these labels do not imply
    # clinical hyper-/hypo-glycemic instability.
    cv_median = result["cv"].median()
    cv_p75 = result["cv"].quantile(0.75)
    cv_clinical = 36.0  # kept as reference column
    mage_median = result["mage"].median()  # cohort-relative MAGE split

    result["label_median"] = (result["cv"] >= cv_median).astype(int)
    result["label_p75"] = (result["cv"] >= cv_p75).astype(int)
    result["label_clinical"] = (result["cv"] >= cv_clinical).astype(int)
    # Secondary endpoint: MAGE cohort-median split.  Days with NaN MAGE
    # (insufficient turning points) are left as NaN, not 0 — dropped in
    # prepare_matrix for any run using this label.
    result["label_mage_median"] = (result["mage"] >= mage_median).astype("Int64")
    result.loc[result["mage"].isna(), "label_mage_median"] = pd.NA

    # Primary label used downstream
    result["label"] = result["label_median"]
    result["label_str"] = result["label"].map({1: "high", 0: "low"})

    # Store thresholds as metadata columns for traceability
    result["cv_threshold_median"] = round(cv_median, 2)
    result["cv_threshold_p75"] = round(cv_p75, 2)
    result["mage_threshold_median"] = round(mage_median, 2)

    # Reorder columns
    cols = [
        "participant", "date", "n_readings",
        "mean_glucose", "sd_glucose", "cv",
        "tir_70_180", "tar_180", "tbr_70", "mage",
        "label", "label_str",
        "label_median", "label_p75", "label_clinical", "label_mage_median",
        "cv_threshold_median", "cv_threshold_p75", "mage_threshold_median",
    ]
    result = result[cols].sort_values(["participant", "date"]).reset_index(drop=True)
    return result, {"cv_median": cv_median, "cv_p75": cv_p75}


if __name__ == "__main__":
    df, thresholds = process_all()

    print("\n--- Summary ---")
    print(f"Total participant-days : {len(df)}")
    print(f"Participants           : {df['participant'].nunique()}")

    print(f"\nCV thresholds used:")
    print(f"  Median (primary)  : {thresholds['cv_median']:.2f}%")
    print(f"  75th percentile   : {thresholds['cv_p75']:.2f}%")
    print(f"  Clinical (ref)    : 36.00%")

    print(f"\nLabel distribution (median split — primary):")
    vc = df["label_str"].value_counts()
    for k, v in vc.items():
        print(f"  {k:5s}: {v:3d}  ({v/len(df)*100:.1f}%)")

    print(f"\nLabel distribution (p75 split — sensitivity):")
    vc2 = df["label_p75"].value_counts()
    for k, v in vc2.items():
        lbl = "high" if k == 1 else "low"
        print(f"  {lbl:5s}: {v:3d}  ({v/len(df)*100:.1f}%)")

    print("\nPer-participant day counts and mean CV:")
    summary = df.groupby("participant").agg(
        n_days=("date", "count"),
        mean_cv=("cv", "mean"),
        pct_high_median=("label_median", "mean"),
    )
    summary["mean_cv"] = summary["mean_cv"].round(2)
    summary["pct_high_median"] = (summary["pct_high_median"] * 100).round(1)
    print(summary.to_string())

    print("\nMetric distributions:")
    print(df[["mean_glucose", "sd_glucose", "cv", "tir_70_180", "mage"]].describe().round(2).to_string())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
