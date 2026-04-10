"""
Wearable Feature Extraction Pipeline
--------------------------------------
Empatica E4 signals used:
  HR   (~1 Hz, minute-level timestamps)   → hr_mean, hr_std, hr_min, hr_max
  IBI  (event-driven, sub-second)         → ibi_mean, ibi_sdnn, ibi_rmssd, ibi_pnn50
  ACC  (32 Hz → downsampled to 1 Hz)      → acc_mean_mag, acc_std_mag, acc_active_pct
  EDA  (4 Hz)                             → eda_mean, eda_std, eda_n_peaks
  TEMP (4 Hz)                             → temp_mean, temp_std, temp_min

BVP is skipped — HR and IBI are already derived from it.

Quality gates (per modality per day):
  HR   : ≥ 21 600 rows  (≥ 6 h at ~1 Hz)
  IBI  : ≥ 100 beats    (minimum for reliable HRV)
  ACC  : ≥ 21 600 s     (≥ 6 h after 1-Hz downsampling)
  EDA  : ≥ 86 400 rows  (≥ 6 h at 4 Hz)
  TEMP : ≥ 86 400 rows  (≥ 6 h at 4 Hz)

Missing modality data for a day → NaN for those features (day not dropped).

Outputs:
  data/wearable_features.csv         — features only, all participant-days
  data/wearable_features_labeled.csv — inner-joined with CGM labels
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")
LABELS_PATH = os.path.join(DATA_DIR, "cgm_daily_labels.csv")
OUT_RAW     = os.path.join(DATA_DIR, "wearable_features.csv")
OUT_LABELED = os.path.join(DATA_DIR, "wearable_features_labeled.csv")

# Quality gate minimums
MIN_HR_ROWS   = 21_600   # 6 h × 3600 s
MIN_IBI_BEATS = 100
MIN_ACC_ROWS  = 21_600   # after 1-Hz downsampling
MIN_EDA_ROWS  = 86_400   # 6 h × 3600 s × 4 Hz
MIN_TEMP_ROWS = 86_400

# EDA SCR peak detection: min prominence (µS), min 4 s between peaks at 4 Hz
EDA_SCR_PROMINENCE = 0.05
EDA_SCR_DISTANCE   = 16   # samples at 4 Hz

# ACC activity: a 1-s epoch is "active" if its magnitude deviates from the
# day's mean by more than this fraction of the day's mean
ACC_ACTIVITY_FRAC = 0.15


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_hr(pid: str) -> pd.DataFrame:
    """HR file: ~1 Hz. Participant 001 uses 'M/D/YY H:MM'; others use ISO format."""
    path = os.path.join(DATA_DIR, pid, f"HR_{pid}.csv")
    df = pd.read_csv(path, header=0, names=["timestamp", "hr"])
    # Try ISO format first (faster); fall back to mixed parsing for participant 001
    ts = pd.to_datetime(df["timestamp"].str.strip(), format="%Y-%m-%d %H:%M:%S", errors="coerce")
    if ts.isna().mean() > 0.5:
        ts = pd.to_datetime(df["timestamp"].str.strip(), format="%m/%d/%y %H:%M", errors="coerce")
    df["timestamp"] = ts
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    df = df.dropna()
    df = df[(df["hr"] >= 30) & (df["hr"] <= 220)]
    df["date"] = df["timestamp"].dt.date
    return df[["date", "hr"]]


def load_ibi(pid: str) -> pd.DataFrame:
    """IBI file: 'YYYY-MM-DD HH:MM:SS.ffffff, ibi_seconds' event-driven."""
    path = os.path.join(DATA_DIR, pid, f"IBI_{pid}.csv")
    df = pd.read_csv(path, header=0, names=["timestamp", "ibi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(), errors="coerce")
    df["ibi"] = pd.to_numeric(df["ibi"], errors="coerce")
    df = df.dropna()
    df = df[(df["ibi"] >= 0.3) & (df["ibi"] <= 2.0)]  # physiological IBI range
    df["date"] = df["timestamp"].dt.date
    return df[["date", "ibi"]]


def load_acc(pid: str) -> pd.DataFrame:
    """
    ACC file: 'YYYY-MM-DD HH:MM:SS.ffffff, x, y, z' at 32 Hz.
    Read in 500k-row chunks, compute vector magnitude, resample to 1 Hz.
    Returns a 1-row-per-second DataFrame with columns [date, mag].
    """
    path = os.path.join(DATA_DIR, pid, f"ACC_{pid}.csv")
    chunks = []
    for chunk in pd.read_csv(
        path, header=0, names=["timestamp", "x", "y", "z"], chunksize=500_000
    ):
        chunk["timestamp"] = pd.to_datetime(
            chunk["timestamp"].str.strip(),
            format="%Y-%m-%d %H:%M:%S.%f",
            errors="coerce",
        )
        chunk = chunk.dropna(subset=["timestamp"])
        chunk["mag"] = np.sqrt(
            chunk["x"].astype(float) ** 2
            + chunk["y"].astype(float) ** 2
            + chunk["z"].astype(float) ** 2
        )
        chunk = chunk.set_index("timestamp")[["mag"]]
        chunk = chunk.resample("1s").mean()
        chunks.append(chunk)

    df = pd.concat(chunks).resample("1s").mean().reset_index()
    df = df.dropna(subset=["mag"])
    df["date"] = df["timestamp"].dt.date
    return df[["date", "mag"]]


def load_eda(pid: str) -> pd.DataFrame:
    """EDA file: 'YYYY-MM-DD HH:MM:SS.fff, eda_µS' at 4 Hz."""
    path = os.path.join(DATA_DIR, pid, f"EDA_{pid}.csv")
    df = pd.read_csv(path, header=0, names=["timestamp", "eda"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(), errors="coerce")
    df["eda"] = pd.to_numeric(df["eda"], errors="coerce")
    df = df.dropna()
    df = df[df["eda"] >= 0]
    df["date"] = df["timestamp"].dt.date
    return df[["date", "eda"]]


def load_temp(pid: str) -> pd.DataFrame:
    """TEMP file: 'YYYY-MM-DD HH:MM:SS.fff, temp_°C' at 4 Hz."""
    path = os.path.join(DATA_DIR, pid, f"TEMP_{pid}.csv")
    df = pd.read_csv(path, header=0, names=["timestamp", "temp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].str.strip(), errors="coerce")
    df["temp"] = pd.to_numeric(df["temp"], errors="coerce")
    df = df.dropna()
    df = df[(df["temp"] >= 20) & (df["temp"] <= 42)]
    df["date"] = df["timestamp"].dt.date
    return df[["date", "temp"]]


# ── Feature computers ─────────────────────────────────────────────────────────

def compute_hr_features(group: pd.DataFrame) -> dict:
    g = group["hr"].values
    if len(g) < MIN_HR_ROWS:
        return {}
    return {
        "hr_mean": round(float(np.mean(g)), 3),
        "hr_std":  round(float(np.std(g, ddof=1)), 3),
        "hr_min":  round(float(np.min(g)), 3),
        "hr_max":  round(float(np.max(g)), 3),
    }


def compute_ibi_features(group: pd.DataFrame) -> dict:
    ibi = group["ibi"].values
    if len(ibi) < MIN_IBI_BEATS:
        return {}
    sdnn  = float(np.std(ibi, ddof=1))
    diffs = np.diff(ibi)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.mean(np.abs(diffs) > 0.05) * 100)
    return {
        "ibi_mean":  round(float(np.mean(ibi)), 4),
        "ibi_sdnn":  round(sdnn, 4),
        "ibi_rmssd": round(rmssd, 4),
        "ibi_pnn50": round(pnn50, 2),
    }


def compute_acc_features(group: pd.DataFrame) -> dict:
    mag = group["mag"].values
    if len(mag) < MIN_ACC_ROWS:
        return {}
    mean_mag   = float(np.mean(mag))
    std_mag    = float(np.std(mag, ddof=1))
    threshold  = mean_mag * ACC_ACTIVITY_FRAC
    active_pct = float(np.mean(np.abs(mag - mean_mag) > threshold) * 100)
    return {
        "acc_mean_mag":   round(mean_mag, 3),
        "acc_std_mag":    round(std_mag, 3),
        "acc_active_pct": round(active_pct, 2),
    }


def compute_eda_features(group: pd.DataFrame) -> dict:
    eda = group["eda"].values
    if len(eda) < MIN_EDA_ROWS:
        return {}
    peaks, _ = find_peaks(
        eda, prominence=EDA_SCR_PROMINENCE, distance=EDA_SCR_DISTANCE
    )
    return {
        "eda_mean":    round(float(np.mean(eda)), 4),
        "eda_std":     round(float(np.std(eda, ddof=1)), 4),
        "eda_n_peaks": int(len(peaks)),
    }


def compute_temp_features(group: pd.DataFrame) -> dict:
    temp = group["temp"].values
    if len(temp) < MIN_TEMP_ROWS:
        return {}
    return {
        "temp_mean": round(float(np.mean(temp)), 3),
        "temp_std":  round(float(np.std(temp, ddof=1)), 3),
        "temp_min":  round(float(np.min(temp)), 3),
    }


# ── Per-participant pipeline ───────────────────────────────────────────────────

MODALITIES = [
    ("HR",   load_hr,   compute_hr_features),
    ("IBI",  load_ibi,  compute_ibi_features),
    ("ACC",  load_acc,  compute_acc_features),
    ("EDA",  load_eda,  compute_eda_features),
    ("TEMP", load_temp, compute_temp_features),
]


def process_participant(pid: str) -> pd.DataFrame:
    day_features: dict[object, dict] = {}  # date → feature dict

    for name, loader, extractor in MODALITIES:
        print(f"    {name}...", end=" ", flush=True)
        try:
            df = loader(pid)
            n_days = 0
            for date, grp in df.groupby("date"):
                feats = extractor(grp)
                if feats:
                    day_features.setdefault(date, {}).update(feats)
                    n_days += 1
            print(f"{n_days} days")
        except FileNotFoundError:
            print("file missing")
        except Exception as e:
            print(f"ERROR: {e}")

    rows = [
        {"participant": pid, "date": date, **feats}
        for date, feats in sorted(day_features.items())
    ]
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    participants = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.isdigit()
    )
    print(f"Processing {len(participants)} participants\n")

    all_frames = []
    for pid in participants:
        print(f"[{pid}]")
        df = process_participant(pid)
        print(f"  → {len(df)} participant-days extracted\n")
        all_frames.append(df)

    features = (
        pd.concat(all_frames, ignore_index=True)
        .sort_values(["participant", "date"])
        .reset_index(drop=True)
    )
    features.to_csv(OUT_RAW, index=False)
    print(f"Saved wearable features → {OUT_RAW}  ({len(features)} rows)")

    # ── Join with CGM labels ──────────────────────────────────────────────────
    labels = pd.read_csv(LABELS_PATH)
    labels["date"] = pd.to_datetime(labels["date"]).dt.date
    labels["participant"] = labels["participant"].astype(str).str.zfill(3)
    features["date"] = pd.to_datetime(features["date"]).dt.date
    features["participant"] = features["participant"].astype(str).str.zfill(3)

    merged = (
        labels.merge(features, on=["participant", "date"], how="inner")
        .sort_values(["participant", "date"])
        .reset_index(drop=True)
    )
    merged.to_csv(OUT_LABELED, index=False)
    print(f"Saved labeled table    → {OUT_LABELED}  ({len(merged)} rows)\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    feat_cols = [
        c for c in merged.columns
        if c not in {
            "participant", "date", "n_readings",
            "mean_glucose", "sd_glucose", "cv",
            "tir_70_180", "tar_180", "tbr_70", "mage",
            "label", "label_str",
            "label_median", "label_p75", "label_clinical",
            "cv_threshold_median", "cv_threshold_p75",
        }
    ]

    print(f"Feature columns ({len(feat_cols)}):")
    print(" ", feat_cols)

    print(f"\nMissing values per feature:")
    missing = merged[feat_cols].isnull().sum()
    for col, n in missing.items():
        pct = n / len(merged) * 100
        flag = " ⚠" if pct > 10 else ""
        print(f"  {col:<20s} {n:3d}  ({pct:.1f}%){flag}")

    print(f"\nFeature distributions:")
    print(merged[feat_cols].describe().round(3).to_string())

    print(f"\nLabel balance in final labeled table:")
    vc = merged["label_str"].value_counts()
    for k, v in vc.items():
        print(f"  {k}: {v} ({v/len(merged)*100:.1f}%)")


if __name__ == "__main__":
    main()
