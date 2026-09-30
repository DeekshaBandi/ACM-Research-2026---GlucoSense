# GlucoSense: Detecting Glycemic Variability from Wearable Sensors

**ACM UT Dallas Research, 2026**

Can a smartwatch-style wearable tell you when your blood sugar is swinging more than usual, without a glucose monitor, finger pricks, or food logs?

GlucoSense tests that idea. We train machine learning models on wrist-worn sensor data (heart rate, heart rate variability, movement, skin conductance, and skin temperature) to predict whether a person had a **high** or **low** glycemic variability day, using continuous glucose monitor (CGM) readings as the ground truth.

The goal is to find out which sensors actually matter, so non-invasive glucose screening could someday run on cheaper, simpler devices.

## Key findings

- **Movement data carries the most signal.** Accelerometer features alone (just 3 features) reached a mean AUROC of **0.725** with Random Forest, beating the full 17-feature model (0.645).
- **Accelerometer + skin temperature was the most consistent pick.** It ranked #1 overall across both models and five metrics (RF AUROC 0.701, XGBoost AUROC 0.686).
- **Heart rate and skin conductance alone did not help.** On their own, they scored at or near chance.
- **All real models beat the chance baseline** (random guessing AUROC: 0.443).
- **Results are early and noisy.** With only 16 participants, fold-to-fold spread is large (AUROC std around 0.2 to 0.3), and results did not hold up well under the stricter labels in the sensitivity analysis. This is a feasibility study, not a finished tool.

## Data

**Source:** [BIG IDEAs Lab Glycemic Variability and Wearable Device Data](https://physionet.org/content/big-ideas-glycemic-wearable/) (PhysioNet, ODC-By license)

- 16 participants wearing a **Dexcom G6 CGM** and an **Empatica E4 wristband** for about 8 to 10 days
- Participants had blood glucose in the normal-to-elevated range (not diagnosed diabetes)
- After quality filtering: **117 participant-days** (112 with every sensor available)

Raw participant folders (`data/001/` to `data/016/`) are not committed. Download them from PhysioNet and place them in `data/`.

## How it works

### 1. Label each day from the CGM (`cgm_processing.py`)

For each participant-day, we compute standard glucose metrics: mean, SD, coefficient of variation (CV), time in range (70 to 180 mg/dL), time above and below range, and MAGE.

The usual clinical cutoff for high variability (CV of 36% or more) is built for people with diabetes. In this group, the average CV is only about 17%, so almost no days would count as "high." Instead we use a **cohort-relative label**:

| Label | Rule | Split | Role |
|---|---|---|---|
| `label_median` | CV at or above the group median | ~50/50 | Primary |
| `label_p75` | CV in the top 25% | ~25/75 | Sensitivity check |
| `label_mage_median` | MAGE at or above the group median | ~50/50 | Secondary check |

### 2. Extract daily wearable features (`wearable_features.py`)

| Sensor | Features |
|---|---|
| Heart rate (HR) | mean, SD, min, max |
| Heart rate variability (IBI) | mean IBI, SDNN, RMSSD, pNN50 |
| Accelerometer (ACC) | mean movement, movement SD, % time active |
| Electrodermal activity (EDA) | mean, SD, number of peaks |
| Skin temperature (TEMP) | mean, SD, min |

Each sensor has a quality gate (for example, at least 6 hours of wear per day). Days missing a sensor keep their other sensors instead of being thrown out.

### 3. Train and evaluate (`train_baseline.py`, `ablation.py`)

- **Models:** Random Forest and XGBoost as co-primary models, Logistic Regression as a simple baseline, and a stratified dummy classifier as the chance floor
- **Validation:** Leave-one-subject-out cross-validation (LOSO). Each fold holds out one whole person, so the model is always tested on someone it has never seen
- **No tuning:** Hyperparameters are fixed sensible defaults. Tuning on 16 people would likely give overly optimistic results
- **Scaling** happens inside each fold to avoid leaking test data
- **Metrics:** AUROC, balanced accuracy, F1, sensitivity, specificity, reported with per-fold spread

### 4. Sensor ablation (`ablation.py`)

We test **17 sensor combinations** (each sensor alone, every pair, every triple, and all together) to see which sensors pull their weight. The "minimum viable" set is the smallest combination where both models stay within 0.05 AUROC of the best result.

### 5. Sensitivity analysis (`sensitivity_and_report.py`)

The top subsets are re-run with the stricter p75 and MAGE labels to check whether the findings hold under different definitions of "high variability."

## Results

### Baseline (all 17 features, LOSO)

| Model | AUROC | Balanced accuracy | F1 |
|---|---|---|---|
| Logistic Regression | 0.570 | 0.438 | 0.341 |
| Random Forest | 0.645 | 0.582 | 0.516 |
| XGBoost | 0.627 | 0.611 | 0.548 |
| Chance (dummy) | 0.443 | | |

### Top sensor combinations (primary label)

| Sensors | # features | RF AUROC | XGB AUROC | Overall rank |
|---|---|---|---|---|
| ACC + TEMP | 6 | 0.701 | 0.686 | 1 |
| ACC | 3 | **0.725** | 0.662 | 2 |
| ACC + EDA + TEMP | 9 | 0.664 | **0.693** | 3 |
| All sensors | 17 | 0.645 | 0.627 | 6 |
| HR only | 4 | 0.496 | 0.462 | 16 |

All values are mean AUROC across LOSO folds. Full tables are in `results/`.

**Minimum viable sensor set:** accelerometer only (3 features).

## Repository structure

```
├── cgm_processing.py          # Step 1: daily glucose metrics + labels
├── wearable_features.py       # Step 2: daily wearable features
├── feature_config.py          # Feature groups and the 17 ablation subsets
├── modeling_utils.py          # Data loading, scaling, LOSO-CV, metrics
├── train_baseline.py          # Step 3: all-feature baseline (LR, RF, XGB)
├── ablation.py                # Step 4: sensor ablation + ensemble + chance floor
├── sensitivity_and_report.py  # Step 5: sensitivity analysis + final summary
├── data/
│   ├── Demographics.csv
│   ├── cgm_daily_labels.csv
│   ├── wearable_features.csv
│   └── wearable_features_labeled.csv
└── results/                   # Metrics, predictions, final config
```

## How to run

```bash
pip install pandas numpy scipy scikit-learn xgboost

# Put raw PhysioNet participant folders in data/001 ... data/016, then:
python cgm_processing.py
python wearable_features.py
python train_baseline.py
python ablation.py
python sensitivity_and_report.py
```

The processed feature files are already in `data/`, so you can skip the first two steps and start from `train_baseline.py`.

## Limitations

- Only 16 participants, so results vary a lot between people
- Labels are relative to this group, not clinical cutoffs
- Participants were not diagnosed with diabetes, so findings may not carry over to people who are
- Findings weakened under the p75 and MAGE labels

## Acknowledgments

Data from the BIG IDEAs Lab at Duke University via PhysioNet.

Bent, B., Cho, P. J., Henriquez, M., et al. BIG IDEAs Lab Glycemic Variability and Wearable Device Data (version 1.1.3). PhysioNet. https://doi.org/10.13026/aw6y-fc44
