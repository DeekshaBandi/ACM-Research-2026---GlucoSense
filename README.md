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

## XGBoost Sensor Ablation (Laasya RF Branch)

**Script:** `xgboost_ablation.py`  
**Data:** `MergedDataset.csv` — 134 daily windows, 16 participants, **no CGM features**  
**Task:** Binary classification — `high` (1) vs `low` (0) glycemic variability  
**Evaluation:** Leave-One-Subject-Out CV (LOSOCV) + SMOTE on each training fold  
**Class distribution:** 82 low / 52 high

F1, Precision, Recall are reported for the **high** (positive) class. ROC-AUC is computed from predicted probabilities.

### Results (sorted by F1)

| Configuration | # Feat | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **IBI only** | 5 | 0.5970 | 0.4773 | 0.4038 | **0.4375** | **0.6399** |
| TEMP only | 4 | 0.5000 | 0.3846 | 0.4808 | 0.4274 | 0.5021 |
| BVP only | 4 | 0.5149 | 0.3934 | 0.4615 | 0.4248 | 0.4906 |
| All wearable modalities | 25 | 0.5522 | 0.4200 | 0.4038 | 0.4118 | 0.5159 |
| ACC only | 4 | 0.5597 | 0.4255 | 0.3846 | 0.4040 | 0.5729 |
| **Baseline (dummy)** | 0 | 0.5000 | 0.3684 | 0.4038 | 0.3853 | 0.4824 |
| HR only | 4 | 0.4851 | 0.3455 | 0.3654 | 0.3551 | 0.4724 |
| EDA only | 4 | 0.4776 | 0.3393 | 0.3654 | 0.3519 | 0.4271 |
| HR + IBI | 9 | 0.5373 | 0.3810 | 0.3077 | 0.3404 | 0.5171 |
| HR + IBI + EDA | 13 | 0.5149 | 0.3556 | 0.3077 | 0.3299 | 0.4852 |
| HR + IBI + ACC | 13 | 0.5373 | 0.3750 | 0.2885 | 0.3261 | 0.4850 |
| HR + ACC | 8 | 0.5149 | 0.3415 | 0.2692 | 0.3011 | 0.4843 |
| HR + EDA | 8 | 0.4925 | 0.3095 | 0.2500 | 0.2766 | 0.4353 |

Full results saved to `results/xgboost_ablation_results.csv`.

### Key Findings

**1. IBI alone is the strongest single modality** — F1 0.44, ROC-AUC 0.64. This is also the only configuration that clearly beats the dummy baseline on AUC, suggesting IBI carries genuine signal about glycemic state. IBI_rmssd (HRV metric) likely drives this; it reflects autonomic nervous system activity which is linked to insulin sensitivity.

**2. Most sensor combinations underperform the dummy baseline on F1.** HR-based combos (HR+IBI, HR+IBI+ACC, HR+IBI+EDA) all score below 0.39 F1, worse than predicting stratified-random. Adding HR to IBI *hurts* performance — HR mean/std are likely noisy at this daily-window granularity.

**3. All wearable modalities combined (F1 0.41) does not beat IBI alone (F1 0.44).** More sensors add noise, not signal, under the current feature set. With 25 features and only 134 samples (16 subjects in LOSOCV), the model overfits training folds.

**4. ACC and TEMP are the second-tier individual sensors** — ACC ROC-AUC 0.57, TEMP recall 0.48. These reflect physical activity and thermoregulation, both relevant to post-meal glucose responses.

**5. EDA and HR alone are weakest** — both below dummy on F1, and EDA ROC-AUC 0.43 is below random chance. The daily-mean EDA feature loses the event-level stress spikes that make EDA useful for glucose prediction.

### Is this approach worth pursuing?

**IBI (and specifically IBI_rmssd / HRV) is the one feature worth building on.** A model using only IBI features achieves the best F1 and the only above-chance AUC in this study. The next step should be to enrich the IBI feature set (e.g. frequency-domain HRV: LF/HF ratio, SDNN) rather than stacking more sensor modalities.

The absolute F1 values (~0.44 best) are still low — this is a hard problem with small data (134 windows, 16 subjects). LOSOCV is the right evaluation protocol here, but the fold sizes are tiny (~8 test windows per subject), making variance high.

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
