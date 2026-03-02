from pathlib import Path

DATA_ROOT = Path("data/big-ideas")
RESAMPLE_RULE = "5min"

LABEL_WINDOW_BINS = 288
LABEL_STD_K = 1.0

N_SPLITS = 3

XGB_PARAMS = dict(
    objective="multi:softprob",
    eval_metric="mlogloss",
    n_estimators=200,
    learning_rate=0.08,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    n_jobs=-1,
    random_state=42,
)


