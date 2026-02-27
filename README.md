# Random Forest Implementation (w/o Food Logs)

This branch documents and implements a glucose excursion detection pipeline based on Dexcom + demographics data, without using food log data.

## Objective

Improve rare excursion detection performance (excursion = hypo or hyper event), while reporting both classification metrics and R2 from predicted probabilities.

- Hypoglycemia threshold: glucose `< 70 mg/dL`
- Hyperglycemia threshold: glucose `> 180 mg/dL`
- Excursion target:
  - `1` if glucose is `< 70` or `> 180`
  - `0` otherwise

## Data Sources Used

- `DESTINATION/*/Dexcom_*.csv` (all 16 participants)
- `DESTINATION/Demographics.csv`
- Food logs are intentionally **not** used.

## Key Code Changes That Produced Current Metrics

### 1) End-to-end multi-participant ingestion

- Load all participant Dexcom files.
- Keep only EGV rows (`Event Type == "EGV"`).
- Parse timestamp and glucose numerically.
- Merge demographics by `participant_id` (`Gender`, `HbA1c`).

### 2) Feature engineering upgrades

Base temporal features:
- `hour`, `minute`, `day_of_week`
- cyclical encoding: `hour_sin`, `hour_cos`

Glucose history features (per participant, shifted to avoid direct lookahead):
- lags: `glucose_lag_1`, `glucose_lag_2`, `glucose_lag_3`, `glucose_lag_6`, `glucose_lag_12`
- rolling means: `glucose_roll_mean_3`, `glucose_roll_mean_6`, `glucose_roll_mean_12`
- rolling stds: `glucose_roll_std_3`, `glucose_roll_std_6`, `glucose_roll_std_12`
- deltas: `glucose_delta_1`, `glucose_delta_3`

### 3) Imbalance-aware modeling

Three models are trained:
- Logistic Regression (`class_weight="balanced"`)
- Random Forest (`class_weight="balanced"`)
- Gradient Boosting (balanced `sample_weight` per fold)

### 4) Group-aware validation (anti-leakage)

- Evaluation uses `LeaveOneGroupOut` with `group = participant_id`.
- This prevents random row-level mixing across participants.

### 5) Threshold optimization for F1

- For each model, out-of-fold probabilities are collected.
- Decision threshold is tuned using precision-recall curve to maximize F1.
- This replaced fixed `0.5` thresholding and is a primary reason F1 improved.

### 6) Reporting additions

For each model, the script prints:
- Precision
- Recall (excursion class)
- F1-score
- R2-score (between true binary labels and predicted probabilities)
- ROC-AUC
- Best threshold (F1-optimized)
- Confusion matrix
- Top feature importances

Also saved:
- ROC plot: `results/excursion_roc_curve.png`

## Current Displayed Results

Dataset:
- Shape: `36,706 x 25`
- Excursion positives: `855 (2.33%)`
- Excursion negatives: `35,851 (97.67%)`
- Participants: `16` (LOGO folds)

Metrics:

1) Logistic Regression (balanced)
- Precision: `0.3947`
- Recall: `0.3813`
- F1: `0.3879`
- R2: `-3.8744`
- ROC-AUC: `0.8310`
- Best threshold: `0.9082`

2) Random Forest (balanced)
- Precision: `0.8916`
- Recall: `0.9041`
- F1: `0.8978`
- R2: `0.8047`
- ROC-AUC: `0.9990`
- Best threshold: `0.3175`

3) Gradient Boosting (balanced sample weights)
- Precision: `0.8673`
- Recall: `0.9099`
- F1: `0.8881`
- R2: `0.7298`
- ROC-AUC: `0.9933`
- Best threshold: `0.8817`

## Run Command

Use this from repo root:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="$(pwd)/.matplotlib" XDG_CACHE_HOME="$(pwd)/.cache" .venv/bin/python randomForestPractice.py
```