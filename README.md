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

> Run the script yourself to generate numbers — output is printed to the terminal.
> Results will vary slightly across machines due to parallelism, but `random_state=42` keeps the split consistent.

The script reports three metrics:

| Metric                             | What it measures                                    | Ideal value             |
| ---------------------------------- | --------------------------------------------------- | ----------------------- |
| **MAE** (Mean Absolute Error)      | Average mg/dL error per prediction                  | Lower is better         |
| **RMSE** (Root Mean Squared Error) | Like MAE but penalises large errors more heavily    | Lower is better         |
| **R² Score**                       | Fraction of glucose variance explained by the model | Closer to 1.0 is better |

The script also prints the **top 10 feature importances** — which features the forest relied on most.

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
│   ├── 001/Dexcom_1.csv
│   ├── ...
│   ├── 016/Dexcom_16.csv
│   └── Demographics.csv
├── randomForestPractice.py       # Vedant's RF regression pipeline
├── results/                      # Output plots
│   └── excursion_roc_curve.png   # ROC curves from classification experiments
└── README.md
```
