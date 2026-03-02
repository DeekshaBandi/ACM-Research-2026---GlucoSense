import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

from .config import DATA_ROOT, XGB_PARAMS, N_SPLITS
from .io_utils import read_signal_csv, read_dexcom_clarity, resample_5min
from .features import core_feature_table, make_labels


def build_participant(pid: str) -> pd.DataFrame:
    pdir = DATA_ROOT / pid

    acc = read_signal_csv(pdir / f"ACC_{pid}.csv")
    bvp = read_signal_csv(pdir / f"BVP_{pid}.csv")
    eda = read_signal_csv(pdir / f"EDA_{pid}.csv")
    hr = read_signal_csv(pdir / f"HR_{pid}.csv")
    ibi = read_signal_csv(pdir / f"IBI_{pid}.csv")
    tmp = read_signal_csv(pdir / f"TEMP_{pid}.csv")
    dex = read_dexcom_clarity(pdir / f"Dexcom_{pid}.csv")

    acc_5 = resample_5min(acc)
    bvp_5 = resample_5min(bvp)
    eda_5 = resample_5min(eda)
    hr_5 = resample_5min(hr)
    ibi_5 = resample_5min(ibi)
    tmp_5 = resample_5min(tmp)
    dex_5 = resample_5min(dex)

    idx = dex_5.index

    acc_5 = acc_5.reindex(idx)
    bvp_5 = bvp_5.reindex(idx)
    eda_5 = eda_5.reindex(idx)
    hr_5 = hr_5.reindex(idx)
    ibi_5 = ibi_5.reindex(idx)
    tmp_5 = tmp_5.reindex(idx)

    food_path = pdir / f"Food_Log_{pid}.csv"
    food_df = pd.read_csv(food_path, skipinitialspace=True) if food_path.exists() else None

    X = core_feature_table(acc_5, bvp_5, eda_5, hr_5, ibi_5, tmp_5, food_df, idx)
    y = make_labels(dex_5)

    df = X.copy()
    df["y"] = y
    df["participant_id"] = pid
    df = df.reset_index().rename(columns={"index": "datetime"})
    return df


def build_dataset() -> pd.DataFrame:
    pids = sorted([p.name for p in DATA_ROOT.iterdir() if p.is_dir() and p.name.isdigit()])
    dfs, failed = [], []

    for pid in pids:
        try:
            dfs.append(build_participant(pid))
        except Exception as e:
            failed.append((pid, repr(e)))

    if failed:
        print("\nParticipants skipped due to errors:")
        for pid, err in failed:
            print(f"  {pid}: {err}")

    if not dfs:
        raise RuntimeError("No participants loaded successfully.")

    return pd.concat(dfs, axis=0, ignore_index=True)


def train_model(data: pd.DataFrame):
    data = data.dropna(subset=["y"]).copy()

    groups = data["participant_id"]
    X = data.drop(columns=["y", "participant_id", "datetime"])
    X = X.replace([np.inf, -np.inf], np.nan)
    y = data["y"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    model = XGBClassifier(**XGB_PARAMS, num_class=len(le.classes_))

    gkf = GroupKFold(n_splits=min(N_SPLITS, groups.nunique()))
    oof = np.zeros(len(X), dtype=int)

    for fold, (tr, te) in enumerate(gkf.split(X, y_enc, groups)):
        w_tr = compute_sample_weight(class_weight="balanced", y=y_enc[tr])
        model.fit(X.iloc[tr], y_enc[tr], sample_weight=w_tr)

        preds = model.predict(X.iloc[te])
        oof[te] = preds

        print(f"Fold {fold}: {balanced_accuracy_score(y_enc[te], preds):.4f}")

    print("Overall:", balanced_accuracy_score(y_enc, oof))
    print(classification_report(y_enc, oof, target_names=le.classes_))
    print(confusion_matrix(y_enc, oof))

    w_all = compute_sample_weight(class_weight="balanced", y=y_enc)
    model.fit(X, y_enc, sample_weight=w_all)
    return model, le


def main():
    data = build_dataset()
    print("Shape:", data.shape)
    print(data["y"].value_counts(dropna=False))
    train_model(data)


if __name__ == "__main__":
    main()

"""
Shape: (38115, 11)
y
PersNorm    24802
PersHigh     5438
PersLow      4930
NaN          2945
Name: count, dtype: int64
Fold 0: 0.4963
Fold 1: 0.5244
Fold 2: 0.4434
Overall: 0.4877101184757591
              precision    recall  f1-score   support

    PersHigh       0.27      0.47      0.35      5438
     PersLow       0.24      0.49      0.32      4930
    PersNorm       0.79      0.51      0.62     24802

    accuracy                           0.50     35170
   macro avg       0.44      0.49      0.43     35170
weighted avg       0.64      0.50      0.53     35170

[[ 2545  1072  1821]
 [ 1057  2405  1468]
 [ 5653  6567 12582]]
 
"""