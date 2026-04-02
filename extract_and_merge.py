# extract_and_merge.py
# Extracts daily HR + IBI features for all 16 participants
# and merges with the CGM label table (merged_de.csv)

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────
DATA_ROOT   = r"C:\Projects\glucosense_data"
MERGED_CGM  = r"C:\Projects\glucosense_data\merged_de.csv"
OUTPUT_PATH = r"C:\Projects\glucosense_data\merged_de_with_wearables.csv"
# ─────────────────────────────────────────────────────────────────────────

def extract_hr_features(participant_id):
    folder = str(participant_id).zfill(3)
    path = os.path.join(DATA_ROOT, folder, f"HR_{folder}.csv")
    if not os.path.exists(path):
        print(f"  [!] HR file missing for participant {participant_id}")
        return pd.DataFrame()

    df = pd.read_csv(path, names=["datetime", "hr"], skiprows=1)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    df = df.dropna(subset=["hr"])

    daily = df.groupby("date")["hr"].agg(
        hr_mean="mean",
        hr_std="std",
        hr_min="min",
        hr_max="max",
        hr_median="median"
    ).reset_index()
    daily["participant"] = participant_id
    return daily

def extract_ibi_features(participant_id):
    folder = str(participant_id).zfill(3)
    path = os.path.join(DATA_ROOT, folder, f"IBI_{folder}.csv")
    if not os.path.exists(path):
        print(f"  [!] IBI file missing for participant {participant_id}")
        return pd.DataFrame()

    df = pd.read_csv(path, names=["datetime", "ibi"], skiprows=1)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"] = df["datetime"].dt.date
    df["ibi"] = pd.to_numeric(df["ibi"], errors="coerce")
    df = df.dropna(subset=["ibi"])

    def rmssd(x):
        diffs = np.diff(x.values)
        return np.sqrt(np.mean(diffs**2)) if len(diffs) > 0 else np.nan

    def pnn50(x):
        diffs = np.abs(np.diff(x.values)) * 1000  # convert to ms
        return (np.sum(diffs > 50) / len(diffs)) * 100 if len(diffs) > 0 else np.nan

    daily = df.groupby("date")["ibi"].agg(
        ibi_mean="mean",
        ibi_std="std",
        ibi_rmssd=rmssd,
        ibi_pnn50=pnn50
    ).reset_index()
    daily["participant"] = participant_id
    return daily

# ── EXTRACT ALL PARTICIPANTS ─────────────────────────────────────────────
print("Extracting wearable features...")
hr_all  = []
ibi_all = []

for pid in range(1, 17):
    print(f"  Processing participant {pid}...")
    hr_all.append(extract_hr_features(pid))
    ibi_all.append(extract_ibi_features(pid))

hr_df  = pd.concat([x for x in hr_all  if not x.empty], ignore_index=True)
ibi_df = pd.concat([x for x in ibi_all if not x.empty], ignore_index=True)

# ── MERGE HR + IBI ───────────────────────────────────────────────────────
hr_df["date"]  = pd.to_datetime(hr_df["date"])
ibi_df["date"] = pd.to_datetime(ibi_df["date"])

wearable_df = pd.merge(hr_df, ibi_df, on=["participant", "date"], how="outer")
print(f"\nWearable features shape: {wearable_df.shape}")

# ── MERGE WITH CGM LABELS ────────────────────────────────────────────────
cgm_df = pd.read_csv(MERGED_CGM)
cgm_df["date"] = pd.to_datetime(cgm_df["date"])

final_df = pd.merge(cgm_df, wearable_df, on=["participant", "date"], how="inner")
print(f"Final merged shape: {final_df.shape}")
print(f"Columns: {final_df.columns.tolist()}")
print(f"\nMissing values:\n{final_df.isnull().sum()}")

final_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")