# src/features/wearable_features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _prep_index(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d = d.dropna(subset=["timestamp"])
    return d.set_index("timestamp").sort_index()


def acc_features_5min(acc_df: pd.DataFrame) -> pd.DataFrame:
    acc = _prep_index(acc_df)

    # magnitude
    mag = np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2).astype("float32")
    acc = acc.assign(mag=mag)

    agg = acc.resample("5min").agg(
        {
            "x": ["mean", "std"],
            "y": ["mean", "std"],
            "z": ["mean", "std"],
            "mag": ["mean", "std"],
        }
    )

    # flatten columns
    agg.columns = [f"acc_{col}_{stat}" for col, stat in agg.columns]
    return agg


def hr_features_5min(hr_df: pd.DataFrame) -> pd.DataFrame:
    hr = _prep_index(hr_df)
    agg = hr["value"].resample("5min").agg(["mean", "std", "min", "max"])
    agg.columns = [f"hr_{c}" for c in agg.columns]
    return agg


def eda_features_5min(eda_df: pd.DataFrame) -> pd.DataFrame:
    eda = _prep_index(eda_df)
    agg = eda["value"].resample("5min").agg(["mean", "std", "max"])
    agg.columns = [f"eda_{c}" for c in agg.columns]
    return agg


def temp_features_5min(temp_df: pd.DataFrame) -> pd.DataFrame:
    temp = _prep_index(temp_df)
    agg = temp["value"].resample("5min").agg(["mean", "std"])
    agg.columns = [f"temp_{c}" for c in agg.columns]
    return agg

