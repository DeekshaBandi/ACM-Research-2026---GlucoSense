import numpy as np


def iter_lopo_splits(df, subject_col="subject_id"):
    subjects = sorted(df[subject_col].unique().tolist())

    for sid in subjects:
        test_mask = df[subject_col] == sid
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        yield sid, train_idx, test_idx


