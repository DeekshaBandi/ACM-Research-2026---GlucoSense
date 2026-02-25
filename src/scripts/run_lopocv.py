from __future__ import annotations

import numpy as np

from src.data.build_table import build_full_table
from src.models.cv import iter_lopo_splits
from src.models.xgb_regression import make_model, rmse, mape


def main():
    data_root = "data/big-ideas"

    df = build_full_table(data_root)

    feature_cols = [c for c in df.columns if c not in ("subject_id", "timestamp", "y")]

    X_all = df[feature_cols].to_numpy(dtype=float)
    y_all = df["y"].to_numpy(dtype=float)

    fold_rmse = []
    fold_mape = []

    for sid, train_idx, test_idx in iter_lopo_splits(df):
        model = make_model()
        model.fit(X_all[train_idx], y_all[train_idx])

        pred = model.predict(X_all[test_idx])

        r = rmse(y_all[test_idx], pred)
        m = mape(y_all[test_idx], pred)

        fold_rmse.append(r)
        fold_mape.append(m)

        print(f"Subject {sid} | RMSE: {r:.2f} | MAPE: {m:.2f}%")

    print("\nOverall Performance")
    print(f"RMSE mean ± std: {np.mean(fold_rmse):.2f} ± {np.std(fold_rmse):.2f}")
    print(f"MAPE mean ± std: {np.mean(fold_mape):.2f}% ± {np.std(fold_mape):.2f}%")


if __name__ == "__main__":
    main()


