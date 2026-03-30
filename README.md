# GlucoSense — ACM Research 2026

Predicting blood glucose levels from continuous glucose monitor (CGM) data using machine learning.

---

## Project Goal

Build a model that predicts a participant's blood glucose level (mg/dL) from time-of-day patterns and basic demographic data. The longer-term goal is to identify glycemic excursions (hypoglycemia and hyperglycemia) without requiring food logs or other manual inputs.

---

## Data

**Source:** [Big Ideas Lab Glycemic Variability Dataset](https://doi.org/10.5683/SP3/AELLPF) — 16 participants wearing Dexcom CGM sensors.

**Files used (stored in `DESTINATION/`):**

- `DESTINATION/<participant_id>/Dexcom_<id>.csv` — raw CGM readings, one file per participant
- `DESTINATION/Demographics.csv` — participant metadata (ID, Gender, HbA1c)

**Key columns extracted from Dexcom files:**
| Column | Description |
|---|---|
| `Timestamp` | Date and time of reading |
| `Event Type` | Filtered to `EGV` (estimated glucose value) rows only |
| `Glucose Value (mg/dL)` | The target variable |

**Participants:** 16 total. Each file may span several days of 5-minute CGM readings.

---

## Preprocessing

Steps performed in `randomForestPractice.py`:

1. **Load & filter** — Read all 16 Dexcom CSVs, keep only `EGV` rows (strips calibration and other event types).
2. **Parse types** — Convert timestamps to `datetime`, glucose to `float`; drop rows where either is missing.
3. **Time features** — Decompose timestamp into `hour`, `minute`, `day_of_week` to avoid passing a raw high-cardinality timestamp to the model.
4. **Merge demographics** — Left-join on `participant_id` to attach `Gender` and `HbA1c` to every reading.
5. **Encode categoricals** — `Gender` is one-hot encoded via `ColumnTransformer`; numeric columns pass through unchanged.
6. **Glycemic flags** (diagnostic only, not model features):
   - `hypoglycemia_factor`: glucose < 70 mg/dL
   - `hyperglycemia_factor`: glucose > 180 mg/dL
7. **Train/test split** — 80/20 random split (`random_state=42`).

**Final feature set fed to the model:**

| Feature          | Type                    |
| ---------------- | ----------------------- |
| `participant_id` | numeric                 |
| `hour`           | numeric (0–23)          |
| `minute`         | numeric (0–59)          |
| `day_of_week`    | numeric (0=Mon … 6=Sun) |
| `Gender`         | one-hot encoded         |
| `HbA1c`          | numeric                 |

---

## Model

**`randomForestPractice.py` — Random Forest Regressor**

Implemented as a scikit-learn `Pipeline` (preprocessing → model):

```python
RandomForestRegressor(
    n_estimators=300,   # 300 decision trees
    max_depth=None,     # trees grow until leaves are pure
    random_state=42,
    n_jobs=-1           # use all CPU cores
)
```

This is a **regression** model — it outputs a continuous glucose prediction in mg/dL, not a class label.

---

## Current Results

Verified by running `randomForestPractice.py` on the full 16-participant dataset (36,898 EGV readings, 80/20 train/test split, `random_state=42`).

**Dataset summary:**
- Total rows: 36,898
- Hypoglycemia events (< 70 mg/dL): 211 (0.57%)
- Hyperglycemia events (> 180 mg/dL): 656 (1.78%)

**Model performance (test set):**

| Metric | Value | What it means |
| ---------------------------------- | ------- | --------------------------------------------------- |
| **MAE** (Mean Absolute Error)      | 6.97 mg/dL | On average, predictions are off by ~7 mg/dL |
| **RMSE** (Root Mean Squared Error) | 12.45 mg/dL | Larger errors are penalised more; gap vs MAE suggests some outlier readings |
| **R² Score**                       | 0.708 | Model explains ~71% of glucose variance |

**Feature importances (ranked):**

| Rank | Feature | Importance |
| ---- | -------------- | ---------- |
| 1 | `hour` | 0.3261 |
| 2 | `day_of_week` | 0.2247 |
| 3 | `minute` | 0.1640 |
| 4 | `participant_id` | 0.1204 |
| 5 | `HbA1c` | 0.0995 |
| 6 | `Gender_FEMALE` | 0.0351 |
| 7 | `Gender_MALE` | 0.0302 |

**Takeaway:** Time-of-day features (`hour`, `minute`, `day_of_week`) dominate, accounting for ~71% of the model's decisions. This suggests glucose follows strong daily rhythms in this dataset. `participant_id` ranking 4th indicates meaningful between-person variation that the model is capturing.

---

## Known Issues with the Current Split

> **These results should be treated as optimistic.** The current evaluation has two leakage problems that inflate MAE, RMSE, and R².

**1. Temporal leakage**
Random shuffling lets the model train on readings from t−5min and t+5min while predicting t. Since glucose changes slowly, it's essentially interpolating between neighbors rather than forecasting. Metrics are inflated.

**2. Participant leakage**
Every participant appears in both train and test. The model learns each person's glucose baseline from their training rows, making their test rows trivially easy to predict. This is why `participant_id` ranks 4th in feature importance.

**Suggested fix:** Sort by timestamp, then take the last 20% as the test set. The model trains on earlier data and predicts genuinely unseen future readings.

**Expected result:** MAE/RMSE go up, R² goes down — not because the model got worse, but because the current numbers were too optimistic.

---

## Sensor Ablation Study

Run with `sensor_ablation.py`. Uses a proper **temporal split** (train on first 80% of readings by time, test on last 20%) and tests 9 configurations of wearable sensor subsets. All models include baseline time + demographic features, so results reflect the *added value* of each sensor group.

**Results (16 participants, temporal split):**

| Configuration | MAE (mg/dL) | RMSE (mg/dL) | R² |
|---|---|---|---|
| Baseline (time + demographics, no wearables) | 20.53 | 27.09 | -0.6863 |
| HR / IBI only | 17.35 | 23.23 | -0.1200 |
| Accelerometer only | 18.01 | 23.69 | -0.1389 |
| EDA only | 17.99 | 23.81 | -0.1505 |
| Temperature only | 19.04 | 25.23 | -0.2911 |
| **HR + Accelerometer** | **16.83** | **22.62** | **-0.0624** |
| HR + EDA | 17.26 | 22.94 | -0.0922 |
| Accelerometer + EDA | 17.82 | 23.55 | -0.1253 |
| All wearable modalities | 17.34 | 23.13 | -0.1106 |

**Key findings:**

1. **All R² values are negative.** A negative R² means the model performs worse than simply predicting the mean glucose every time. This is the honest result from a proper temporal split — future glucose is genuinely hard to predict from these features alone, and Vedant's R²=0.71 was an artifact of data leakage.

2. **HR + Accelerometer is the best combination** — lowest MAE (16.83 mg/dL) and closest R² to zero (-0.06). Motion and heart rate together carry the most signal.

3. **HR / IBI alone is the best single modality** — MAE=17.35, meaningfully better than temperature (19.04) and the no-sensor baseline (20.53).

4. **Temperature adds little** — worst single sensor and doesn't improve any multi-sensor combo it's added to. "All modalities" (22 features) underperforms "HR + Accelerometer" (18 features).

5. **The baseline (time + demographics) performs worst of all** — MAE=20.53, R²=-0.69. Without sensor data, time-of-day patterns alone do not generalize to future readings.

**Is the original Vedant approach still worth keeping?**

Yes, as a **baseline and comparison track**, but not as a standalone result. It shows the upper bound of what time + demographic features can do *when leakage is present*, and serves as a reference point for future improvements. Any new model should be benchmarked against the honest temporal-split baseline (MAE=20.53) rather than the leaked R²=0.71 number.

---

## Metric Explanations

### MAE — Mean Absolute Error

The average absolute difference between predicted and actual glucose values. An MAE of 15 means the model is off by ~15 mg/dL on average. Easy to interpret in clinical units.

### RMSE — Root Mean Squared Error

Similar to MAE but squares each error before averaging, so one large miss (e.g. 80 mg/dL off) hurts the score much more than many small ones. RMSE ≥ MAE; a large gap between them means the model has occasional bad outliers.

### R² Score

A value between 0 and 1 (can go negative for very bad models). R² = 0.9 means the model explains 90% of the variance in glucose readings. R² = 0 means the model is no better than predicting the mean glucose every time.

### F1 Score _(for future classification work)_

F1 is the harmonic mean of **Precision** and **Recall**:

```
Precision = TP / (TP + FP)   → of all times we predicted hypoglycemia, how often were we right?
Recall    = TP / (TP + FN)   → of all actual hypoglycemia events, how many did we catch?
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```

F1 is the right metric when classes are **imbalanced** — e.g. hypoglycemia events are rare, so a model that always predicts "normal" would get high accuracy but F1 ≈ 0 because it catches nothing. A high F1 (close to 1.0) means the model is both precise and comprehensive.

---

## How to Run

```bash
# Install dependencies
pip install pandas scikit-learn numpy

# Run Vedant's regression pipeline
python randomForestPractice.py
```

The script auto-discovers all `DESTINATION/*/Dexcom_*.csv` files relative to its location — no path changes needed as long as your `DESTINATION/` folder is in the same directory.

---

## Repository Structure

```
ACM-Research-2026---GlucoSense/
├── DESTINATION/                  # Raw dataset (not committed)
│   ├── 001/
│   │   ├── Dexcom_001.csv        # CGM glucose readings
│   │   ├── HR_001.csv            # Heart rate (1 Hz)
│   │   ├── IBI_001.csv           # Inter-beat interval (irregular)
│   │   ├── ACC_001.csv           # Accelerometer x/y/z (32 Hz)
│   │   ├── EDA_001.csv           # Electrodermal activity (4 Hz)
│   │   ├── TEMP_001.csv          # Skin temperature (4 Hz)
│   │   └── BVP_001.csv           # Blood volume pulse (64 Hz)
│   ├── 002/ … 016/
│   └── Demographics.csv
├── randomForestPractice.py       # Vedant's original RF regression (Dexcom + demographics only)
├── sensor_ablation.py            # Sensor ablation across wearable modalities (temporal split)
├── results/
│   ├── sensor_ablation_results.csv
│   └── excursion_roc_curve.png
└── README.md
```
