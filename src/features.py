import numpy as np
import pandas as pd
from .config import LABEL_WINDOW_BINS, LABEL_STD_K


def _food_time_index(food_df: pd.DataFrame) -> pd.Series | None:
    f = food_df.copy()
    f.columns = f.columns.str.strip().str.lower()

    if "time_begin" in f.columns:
        return pd.to_datetime(f["time_begin"], errors="coerce", format="mixed")

    if "date" in f.columns and ("time" in f.columns or "time_of_day" in f.columns):
        tcol = "time" if "time" in f.columns else "time_of_day"
        return pd.to_datetime(
            f["date"].astype(str) + " " + f[tcol].astype(str),
            errors="coerce",
            format="mixed",
        )

    return None


def food_core_features(food_df: pd.DataFrame | None, index: pd.DatetimeIndex) -> pd.DataFrame:
    feats = pd.DataFrame(index=index)

    if food_df is None or food_df.empty:
        feats["carbs_3h"] = 0.0
        feats["time_since_last_meal_min"] = np.inf
        return feats

    f = food_df.copy()
    f.columns = f.columns.str.strip().str.lower()
    dt = _food_time_index(f)

    if dt is None:
        feats["carbs_3h"] = 0.0
        feats["time_since_last_meal_min"] = np.inf
        return feats

    f["__t"] = dt
    f = f.dropna(subset=["__t"]).sort_values("__t").set_index("__t")

    carb_col = None
    for cand in ["total_carb", "total carb", "carb", "carbs"]:
        if cand in f.columns:
            carb_col = cand
            break

    if carb_col is None:
        feats["carbs_3h"] = 0.0
        feats["time_since_last_meal_min"] = np.inf
        return feats

    carbs = pd.to_numeric(f[carb_col], errors="coerce").fillna(0.0)
    carbs_5 = carbs.resample("5min").sum().reindex(index).fillna(0.0)
    feats["carbs_3h"] = carbs_5.rolling(36, min_periods=1).sum()

    meal_times = f.index.values
    idx = index.values
    pos = np.searchsorted(meal_times, idx, side="right") - 1
    last_meal = np.where(pos >= 0, meal_times[pos], np.datetime64("NaT"))
    last_meal = pd.to_datetime(last_meal)

    feats["time_since_last_meal_min"] = (pd.Series(index, index=index) - last_meal).dt.total_seconds() / 60.0
    feats["time_since_last_meal_min"] = feats["time_since_last_meal_min"].fillna(np.inf)

    return feats


def core_feature_table(acc_5, bvp_5, eda_5, hr_5, ibi_5, tmp_5, food_df, idx) -> pd.DataFrame:
    acc_mag = np.sqrt(acc_5["acc_x"] ** 2 + acc_5["acc_y"] ** 2 + acc_5["acc_z"] ** 2)

    X = pd.DataFrame(
        {
            "hr": hr_5["hr"],
            "eda": eda_5["eda"],
            "temp": tmp_5["temp"],
            "bvp": bvp_5["bvp"],
            "ibi": ibi_5["ibi"],
            "acc_mag": acc_mag,
        },
        index=idx,
    )

    X = pd.concat([X, food_core_features(food_df, idx)], axis=1)
    return X


def make_labels(dex_5: pd.DataFrame) -> pd.Series:
    g = dex_5["glucose"]
    past = g.shift(1)
    mu = past.rolling(LABEL_WINDOW_BINS, min_periods=LABEL_WINDOW_BINS // 3).mean()
    sd = past.rolling(LABEL_WINDOW_BINS, min_periods=LABEL_WINDOW_BINS // 3).std()

    y = pd.Series(index=g.index, dtype="object")
    y[g > (mu + LABEL_STD_K * sd)] = "PersHigh"
    y[g < (mu - LABEL_STD_K * sd)] = "PersLow"
    y[(g <= (mu + LABEL_STD_K * sd)) & (g >= (mu - LABEL_STD_K * sd))] = "PersNorm"
    y[(mu.isna()) | (sd.isna())] = np.nan
    return y
