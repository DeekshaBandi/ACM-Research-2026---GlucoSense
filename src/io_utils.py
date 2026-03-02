from pathlib import Path
import pandas as pd
from .config import RESAMPLE_RULE


def read_signal_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", format="mixed")
    df = df.dropna(subset=["datetime"]).sort_values("datetime").set_index("datetime")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_dexcom_clarity(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    tcol = "Timestamp (YYYY-MM-DDThh:mm:ss)"
    etype = "Event Type"
    gcol = "Glucose Value (mg/dL)"

    df = df[df[etype].astype(str).str.strip().eq("EGV")].copy()
    df[tcol] = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
    df[gcol] = pd.to_numeric(df[gcol], errors="coerce")
    df = df.dropna(subset=[tcol, gcol]).sort_values(tcol)

    df = df[[tcol, gcol]].rename(columns={tcol: "datetime", gcol: "glucose"}).set_index("datetime")
    df.index = df.index.floor(RESAMPLE_RULE)
    df = df.groupby(df.index)[["glucose"]].mean()
    return df


def resample_5min(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    if df.empty:
        return df
    if how == "mean":
        return df.resample(RESAMPLE_RULE).mean()
    if how == "sum":
        return df.resample(RESAMPLE_RULE).sum()
    if how == "last":
        return df.resample(RESAMPLE_RULE).last()
    raise ValueError


