import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="CGM daily variabiilty labeling pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("../data"), help="Path to Dexcom CSV folder")
    parser.add_argument("--output", type=Path, default=Path("../output/cgm_daily_labels.csv"), help="Output CSV path")
    parser.add_argument("--min-days", type=int, default=2, help="Minimum days per participant to keep")
    parser.add_argument("--label-method", choices=["clinical", "median"], default="clinical", help="How to binarize CV")
    parser.add_argument("--cv-threshold", type=float, default=36.0, help="Clinical CV threshold (percent) for high variability")
    return parser.parse_args()


def load_dexcom_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    if "Timestamp (YYYY-MM-DDThh:mm:ss)" not in df.columns:
        raise ValueError(f"Unrecognized Dexcom CSV schema: {path}")

    df = df.rename(columns={
        "Timestamp (YYYY-MM-DDThh:mm:ss)": "timestamp",
        "Glucose Value (mg/dL)": "glucose_mg_dl",
        "Event Type": "event_type"
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df[df["timestamp"].notna()]

    if "event_type" in df.columns:
        # Retain actual glucose values, not alerts/metadata.
        df = df[df["event_type"].isin(["EGV", "fg", "Trend", ""], ) | df["event_type"].isna()]

    df["glucose_mg_dl"] = pd.to_numeric(df["glucose_mg_dl"], errors="coerce")
    df = df[df["glucose_mg_dl"].notna()]
    df = df.sort_values("timestamp")

    return df


def compute_mage(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) < 3:
        return np.nan

    sd = values.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return np.nan

    arr = values.to_numpy()
    extrema_idx = [0]
    for i in range(1, len(arr) - 1):
        if (arr[i] > arr[i - 1] and arr[i] >= arr[i + 1]) or (arr[i] < arr[i - 1] and arr[i] <= arr[i + 1]):
            extrema_idx.append(i)
    extrema_idx.append(len(arr) - 1)

    amps = []
    for i in range(1, len(extrema_idx)):
        amp = abs(arr[extrema_idx[i]] - arr[extrema_idx[i - 1]])
        if amp >= sd:
            amps.append(amp)

    if len(amps) == 0:
        return np.nan
    return float(np.mean(amps))


def daily_summary_metrics(daily_df: pd.DataFrame) -> pd.DataFrame:
    day_groups = daily_df.groupby(daily_df["timestamp"].dt.date)
    rows = []

    for day, group in day_groups:
        values = group["glucose_mg_dl"].dropna().astype(float)
        if values.empty:
            continue

        mean = float(values.mean())
        sd = float(values.std(ddof=0))
        cv = float(100.0 * sd / mean) if mean > 0 else np.nan
        mage = compute_mage(values)

        tir = float(((values >= 70) & (values <= 180)).mean() * 100)
        tar = float((values > 180).mean() * 100)
        tbr = float((values < 70).mean() * 100)

        rows.append({
            "date": pd.to_datetime(day).date(),
            "n_readings": len(values),
            "mean_glucose": mean,
            "sd_glucose": sd,
            "cv_percentage": cv,
            "mage": mage,
            "tir_percent": tir,
            "tar_percent": tar,
            "tbr_percent": tbr,
            "min_glucose": float(values.min()),
            "max_glucose": float(values.max()),
        })

    return pd.DataFrame(rows)


def label_variability(df: pd.DataFrame, method: str, clinical_threshold: float) -> pd.DataFrame:
    df = df.copy()
    if method == "clinical":
        threshold = clinical_threshold
    elif method == "median":
        threshold = float(df["cv_percentage"].median())
    else:
        raise ValueError("Unsupported label method")

    df["cv_threshold_used"] = threshold
    df["label_cv_threshold"] = "low"
    df.loc[df["cv_percentage"] >= threshold, "label_cv_threshold"] = "high"
    df["label_cv_numeric"] = (df["cv_percentage"] >= threshold).astype(int)
    return df


def collect_all_days(data_dir: Path, min_days: int):
    all_rows = []
    subject_counts = []

    for csv_path in sorted(data_dir.glob("Dexcom_*.csv")):
        subject_id = csv_path.stem.split("_")[-1]
        try:
            raw = load_dexcom_file(csv_path)
        except Exception as e:
            print(f"Skipped {csv_path}: {e}")
            continue

        if raw.empty:
            print(f"Skipped subject {subject_id}: no usable CGM data")
            continue

        daily = daily_summary_metrics(raw)
        if daily.empty:
            print(f"Skipped subject {subject_id}: no daily summary after aggregation")
            continue

        subject_counts.append({"subject_id": subject_id, "n_days": len(daily)})
        if len(daily) < min_days:
            print(f"Subject {subject_id} has only {len(daily)} days (<min_days={min_days}), excluding from final dataset")
            continue

        daily["subject_id"] = subject_id
        all_rows.append(daily)

    participants_summary = pd.DataFrame(subject_counts)
    if participants_summary.empty:
        raise RuntimeError("No subject produced daily summaries")

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    return combined, participants_summary


def main():
    args = parse_args()
    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined, participants_summary = collect_all_days(data_dir, args.min_days)
    if combined.empty:
        raise RuntimeError("No participant-day rows after filtering")

    print("Participant-day counts:")
    print(participants_summary.sort_values("n_days"))

    labeled = label_variability(combined, method=args.label_method, clinical_threshold=args.cv_threshold)

    # sort by subject and date to keep reproducibility
    labeled = labeled.sort_values(["subject_id", "date"]).reset_index(drop=True)

    output_path.write_text(labeled.to_csv(index=False))
    print(f"Saved labeled participant-day table to {output_path}")
    print(labeled["label_cv_threshold"].value_counts(dropna=False))

    if args.label_method == "clinical":
        high = labeled.loc[labeled["label_cv_threshold"] == "high", "subject_id"].nunique()
        low = labeled.loc[labeled["label_cv_threshold"] == "low", "subject_id"].nunique()
        print(f"Subjects with high CV days: {high}, low CV days: {low}")


if __name__ == "__main__":
    main()
