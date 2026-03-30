# GlucoSense — Laasya's Branch

## Project Goal

Predict whether a person's blood glucose is currently in a **persistently high**, **persistently low**, or **normal** state using wearable physiological signals and food log data — without a continuous glucose monitor (CGM) as input at inference time. The long-term aim is a non-invasive glucose excursion detector.

---

## Data

**Dataset:** [Big Ideas Lab Glycemic Variability Dataset](https://physionet.org/content/big-ideas-glycemic-variability/1.1.1/) (16 participants)

Each participant folder contains:
| File | Signal | Native rate |
|---|---|---|
| `ACC_<pid>.csv` | 3-axis accelerometer (x, y, z) | 32 Hz |
| `BVP_<pid>.csv` | Blood volume pulse | 64 Hz |
| `EDA_<pid>.csv` | Electrodermal activity | 4 Hz |
| `HR_<pid>.csv` | Heart rate | 1 Hz |
| `IBI_<pid>.csv` | Inter-beat interval | ~1 Hz |
| `TEMP_<pid>.csv` | Skin temperature | 4 Hz |
| `Dexcom_<pid>.csv` | CGM glucose readings (Dexcom Clarity) | ~5 min |
| `Food_Log_<pid>.csv` | Self-reported meals with carb counts | event-based |
| `Demographics.csv` | Age, sex, BMI per participant | — |

Data is placed under `data/big-ideas/<participant_id>/`. The path is configured in [src/config.py](src/config.py).

---

## Preprocessing

All steps are implemented in [src/io_utils.py](src/io_utils.py) and [src/features.py](src/features.py).

**1. Resampling to 5-minute bins**
All signals are resampled to a uniform 5-minute grid using `mean` aggregation (`resample_5min` in `io_utils.py`). This aligns the high-frequency wearable data with the CGM's native ~5-minute cadence.

**2. Glucose (CGM) parsing**
Only rows with `Event Type == "EGV"` (Estimated Glucose Value) are kept. Timestamps are floored to the nearest 5-minute bin, and duplicate bins are averaged.

**3. Feature construction** (per 5-minute bin)
| Feature | Description |
|---|---|
| `hr` | Mean heart rate (bpm) |
| `eda` | Mean electrodermal activity (µS) |
| `temp` | Mean skin temperature (°C) |
| `bvp` | Mean blood volume pulse |
| `ibi` | Mean inter-beat interval (s) |
| `acc_mag` | Euclidean magnitude of ACC: √(x²+y²+z²) |
| `carbs_3h` | Rolling sum of carbohydrates consumed in the past 3 hours (g) |
| `time_since_last_meal_min` | Minutes elapsed since the most recent logged meal |

**4. Label generation** (`make_labels` in `features.py`)
Labels are computed relative to each participant's own rolling glucose history (24-hour window = 288 five-minute bins):

- **PersHigh** — current glucose > rolling mean + 1 SD
- **PersLow** — current glucose < rolling mean − 1 SD
- **PersNorm** — glucose within ±1 SD of rolling mean
- **NaN** — insufficient history (first ~24 hours per participant, dropped before training)

This relative labeling accounts for inter-individual differences in baseline glucose.

---

## Model

**Algorithm:** XGBoost Classifier (`XGBClassifier`, `multi:softprob` objective)

**Validation:** Group K-Fold cross-validation (3 folds), grouped by `participant_id` so no participant appears in both train and test. This tests generalization to *unseen individuals*.

**Class imbalance handling:** Sample weights (`compute_sample_weight("balanced")`) applied during training to counteract the ~5:1 ratio of PersNorm to excursion classes.

Key hyperparameters (see [src/config.py](src/config.py)):
```
n_estimators=200, learning_rate=0.08, max_depth=4,
subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0
```

---

## Current Results

Dataset size: **38,115 rows × 8 features**, 16 participants

Class distribution (after label generation):
| Class | Count |
|---|---|
| PersNorm | 24,802 |
| PersHigh | 5,438 |
| PersLow | 4,930 |
| NaN (dropped) | 2,945 |

Cross-validation performance:
| Fold | Balanced Accuracy |
|---|---|
| 0 | 0.4095 |
| 1 | 0.4467 |
| 2 | 0.4193 |
| **Overall** | **0.4267** |

Per-class metrics (out-of-fold predictions):
| Class | Precision | Recall | F1 |
|---|---|---|---|
| PersHigh | 0.18 | 0.52 | 0.27 |
| PersLow | 0.23 | 0.45 | 0.30 |
| PersNorm | 0.79 | 0.31 | 0.45 |
| **Macro avg** | 0.40 | 0.43 | **0.34** |

Confusion matrix (rows = true, columns = predicted — High / Low / Norm):
```
[[ 2820  1299  1319],
 [ 1922  2212   796],
 [10855  6186  7761]]
```

**Takeaway:** The model detects excursions at above-random recall (~0.43 balanced accuracy vs. 0.33 baseline), but precision is very low for both excursion classes — meaning most predicted excursions are false alarms. PersNorm recall collapses to 0.31 because the model over-predicts excursions when class weights are applied. Macro F1 of 0.34 is the primary target to improve.

---

## Metric Explanations

**Accuracy** — fraction of all predictions that are correct. Misleading here because ~70% of data is PersNorm; a model that always predicts PersNorm scores 70% accuracy while being useless for detecting excursions.

**Balanced accuracy** — average recall across all classes. A random 3-class classifier scores 0.33; this model scores ~0.49.

**Precision** — of all times the model predicted class X, how often was it right. Low precision for excursion classes means many false alarms.

**Recall** — of all true class X instances, how many did the model catch. Critical for medical use — missing a real low-glucose event is costly.

**F1 score** — harmonic mean of precision and recall:
`F1 = 2 × (precision × recall) / (precision + recall)`
It balances both concerns. A macro F1 averages F1 equally across all classes, giving equal weight to rare excursion classes regardless of their size. This is the primary metric to optimize when class imbalance is present.

---

## Code Structure

```
src/
  config.py      — data path, resampling rule, label parameters, XGBoost hyperparameters
  io_utils.py    — CSV readers and 5-minute resampler
  features.py    — feature table construction and label generation
  train.py       — dataset builder, cross-validation loop, result printing
```

To run:
```bash
python -m src.train
```
