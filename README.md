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

## Sensor Ablation Study

Each condition uses the same XGBoost + Group K-Fold setup. Food-log features (`carbs_3h`, `time_since_last_meal_min`) are included in all conditions **except** `all_wearable_no_food`, which tests sensors alone.

Results sorted by macro F1:

| Condition | Features | Balanced Acc | Macro F1 | F1 PersHigh | F1 PersLow | F1 PersNorm |
|---|---|---|---|---|---|---|
| all_wearable_no_food | HR, IBI, EDA, Temp, BVP, ACC | **0.4259** | **0.3403** | 0.2660 | 0.3033 | 0.4514 |
| all_wearable | HR, IBI, EDA, Temp, BVP, ACC + food | 0.4236 | 0.3384 | 0.2654 | 0.3003 | 0.4494 |
| temp_only | Temp + food | 0.3812 | 0.3159 | 0.2248 | 0.2565 | 0.4663 |
| hr_ibi_acc | HR, IBI, ACC + food | 0.4119 | 0.3142 | 0.2565 | 0.2871 | 0.3990 |
| hr_ibi_eda | HR, IBI, EDA + food | 0.4118 | 0.3115 | 0.2598 | 0.2921 | 0.3827 |
| acc_eda | ACC, EDA + food | 0.3953 | 0.3064 | 0.2508 | 0.2741 | 0.3942 |
| hr_ibi_only | HR, IBI + food | 0.4127 | 0.3062 | 0.2584 | 0.2850 | 0.3750 |
| acc_only | ACC + food | 0.3781 | 0.3053 | 0.2191 | 0.2662 | 0.4307 |
| bvp_only | BVP + food | 0.3481 | 0.2986 | 0.2423 | 0.1851 | 0.4683 |
| eda_only | EDA + food | 0.3677 | 0.2663 | 0.2626 | 0.2199 | 0.3165 |

Run with: `python sensor_ablation.py`. Raw results saved to `results/sensor_ablation.json`.

### Key Findings

**1. All sensors together is best — but only marginally.**
`all_wearable` (macro F1 0.338) barely outperforms `hr_ibi_acc` (0.314) and `hr_ibi_eda` (0.312). Adding more sensors yields diminishing returns in this linear-feature regime.

**2. Food log features add essentially nothing.**
`all_wearable_no_food` (F1 0.340) slightly *outperforms* `all_wearable` (F1 0.338). The carbohydrate rolling sum and time-since-meal features do not improve excursion detection over sensors alone under the current feature engineering. This is likely because: (a) food logs are incomplete/self-reported, and (b) the 5-minute mean aggregation flattens the post-meal glucose spike dynamics.

**3. HR + IBI is the most efficient single-modality pair.**
`hr_ibi_only` (F1 0.306, balanced acc 0.413) performs comparably to the full sensor set despite using only 2 signals (+ food). It achieves higher balanced accuracy than most combinations. This makes it a strong candidate for a reduced-sensor deployment.

**4. BVP and EDA are the weakest individual modalities.**
`bvp_only` (F1 0.299) and `eda_only` (F1 0.266) perform near or below random on excursion detection individually. EDA's poor solo performance is notable given its use in stress detection; it may require richer temporal features (e.g., SCR peaks) rather than 5-minute means.

**5. Temperature is surprisingly competitive as a single sensor.**
`temp_only` achieves macro F1 0.316 — higher than HR+IBI and most multi-sensor combinations — but this is largely driven by PersNorm F1 (0.47), not excursion detection. Its balanced accuracy (0.381) is lower than HR+IBI, meaning it misses more true excursions.

### Is the Original Approach Worth Pursuing?

**Yes, as a baseline — but it has a hard ceiling with current features.**

The original `all_wearable` XGBoost model (macro F1 0.34, balanced acc 0.43) is a valid baseline for comparing against future improvements. The split is methodologically sound (no participant leakage). However, the ablation reveals that the performance gap between using 2 sensors and 6 sensors is only ~0.03 F1 points, which strongly suggests the **bottleneck is feature representation, not sensor coverage**.

Recommended next steps that are likely to move the needle more than adding sensors:
- **Temporal features**: rolling statistics (mean, SD, slope) over 15/30/60-minute windows per sensor, rather than raw 5-minute means
- **HRV features**: RMSSD, SDNN computed from IBI rather than mean IBI
- **EDA peak detection**: SCR count and amplitude rather than tonic mean
- **Post-meal dynamics**: glucose rate-of-change (ΔG/Δt) or lagged sensor features relative to meal events

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
  config.py           — data path, resampling rule, label parameters, XGBoost hyperparameters
  io_utils.py         — CSV readers and 5-minute resampler
  features.py         — feature table construction and label generation
  train.py            — dataset builder, cross-validation loop, result printing
sensor_ablation.py    — sensor subset ablation study (produces results/sensor_ablation.json)
results/
  sensor_ablation.json — ablation results (all conditions, all metrics)
```

Run baseline:
```bash
python -m src.train
```

Run sensor ablation:
```bash
python sensor_ablation.py
```
