# GlucoSense Pipeline Audit Report
*Study guide for the research presentation — co-author / presenter reference*

This document is a plain-English walk-through of the GlucoSense sensor-ablation pipeline as it exists on the `harmonized-primary-analysis` branch. It is meant as a presentation study guide: read it end-to-end to get the big picture, then use the section headings as quick reference while rehearsing answers to audience questions. The numbers and branch names below match the current agreed state of the project. If they don't, the code is right and this document is wrong — fix it.

---

## 1. What we are actually asking

**Research question.** In a small cohort of people who *do not have diabetes*, can we classify each person's daily glycemic variability (how much blood glucose swings that day) from simple wearable sensor features alone, and if so, what is the **smallest** combination of wearable sensors that still does a reasonable job?

**Why this matters.** Continuous glucose monitors (CGMs) are the gold standard for glycemic variability, but they are expensive, invasive, and not available to most people. Consumer wearables (smartwatches, fitness bands) are cheap and everywhere. If a small subset of wearable signals is enough to flag days with higher vs lower glycemic variability in a healthy cohort, that is a **feasibility signal** that the same approach might eventually help people monitor glycemic health without a CGM strapped to their arm.

**What "minimum viable sensor configuration" means here.** Not "the smartest model" or "the highest AUROC." It means: **the smallest subset of the Empatica E4's five sensor modalities that performs comparably to fuller sensor sets without a clear loss in discrimination.** Fewer sensors = lower burden, cheaper devices, simpler integration. The prize for answering this question well is a concrete shopping list for a next-generation low-burden monitor.

---

## 2. Current architecture at a glance

| Layer | What happens | Key artifact |
|---|---|---|
| **CGM processing** | Dexcom glucose values per participant → daily summary metrics + labels | `data/cgm_daily_labels.csv` |
| **Wearable features** | Empatica E4 raw streams (HR, IBI, ACC, EDA, TEMP) → daily features | `data/wearable_features_labeled.csv` |
| **Config / feature groups** | Single source of truth for labels, sensor modalities, subsets | `feature_config.py` |
| **Modeling utilities** | Data loading, matrix prep, LOSO-CV runner, fold metrics | `modeling_utils.py` |
| **Full-feature baseline** | LR + RF + XGB on all 17 features | `train_baseline.py` |
| **Main ablation** (main-paper pipeline) | RF + XGB + Dummy on all 17 subsets, against the primary endpoint | `ablation.py` |
| **Sensitivity + final report** | MAGE and p75 sensitivity re-runs + Step-12 interpretation | `sensitivity_and_report.py` |
| **Reporting strategy** | Canonical co-author reference for primary/sensitivity roles | `REPORTING_STRATEGY.md` |

---

## 3. Pipeline flow: raw data → final results

### Step 1 — CGM daily labels (`cgm_processing.py`)
For each participant, for each day, compute: mean glucose, SD, **coefficient of variation (CV)**, time-in-range, time-above/below-range, and **MAGE** (Mean Amplitude of Glycemic Excursions — average size of the day's big glucose swings). Days with fewer than 144 CGM readings (<50% of a 24h 5-minute-interval trace) are dropped for quality. Cohort-relative binary labels are then attached:
- `label_median` = 1 if day's CV ≥ cohort median CV — **this is the pre-registered primary label.**
- `label_p75` = 1 if day's CV ≥ 75th percentile of cohort CV (sensitivity label).
- `label_mage_median` = 1 if day's MAGE ≥ cohort median MAGE (sensitivity label).
- `label_clinical` = 1 if day's CV ≥ 36% (AACE clinical cutoff — reference column only; this cohort is normoglycemic so the clinical cutoff is effectively unused).

Output: `data/cgm_daily_labels.csv`, one row per participant-day, with glucose metrics + all four labels.

### Step 2 — Wearable feature extraction
For the same participant-days, extract daily summary features from Empatica E4 raw streams (HR at ~1 Hz, IBI event-driven, ACC at 32 Hz → 1 Hz, EDA at 4 Hz, TEMP at 4 Hz). The full feature list is 17 numeric features grouped into five sensor modalities — see §4 below. Output: `data/wearable_features_labeled.csv`, one row per participant-day, merged with the labels from Step 1.

### Step 3 — Feature groups and subsets (`feature_config.py`)
Defines the five modality feature groups, a paired `hr_ibi` group (HR and IBI both come from the same PPG sensor on the E4, so they count as one physical sensor for ablation), and **17 ablation subsets**: 5 singles + 1 paired (`hr_ibi`) + 6 pairs + 4 triplets + 1 full set. This module is imported everywhere, so all scripts see the same label hierarchy and the same subset definitions.

### Step 4 — Modeling utilities (`modeling_utils.py`)
Provides the shared machinery used by both the baseline and the ablation:
- `load_data()` — reads `wearable_features_labeled.csv`.
- `prepare_matrix(df, feats, label_col)` — drops rows missing the chosen label, returns `X`, `y`, `parts` (participant IDs).
- `run_loso_cv(clf, X, y, parts, scale)` — leave-one-subject-out loop. For each participant, trains on the other 15 and predicts on the held-out participant's rows. If `scale=True`, fits a StandardScaler *inside* the fold (training fold only, then applied to test fold — no leakage). For XGBoost, `scale_pos_weight` is recomputed from the training-fold class ratio each fold, so we never leak the test-fold class distribution into the training weights.
- `compute_fold_metrics(preds)` — aggregates per-fold predictions into: per-fold AUROC mean + std, fold-SE 95% CI, pooled OOF AUROC, **participant-stratified (cluster) bootstrap** CI on pooled AUROC, balanced accuracy, macro-F1, sensitivity, specificity, and `auroc_folds_eval` (how many of the 16 LOSO folds were non-degenerate).

### Step 5 — Full-feature baseline (`train_baseline.py`)
Runs LR, RF, XGB on the complete 17-feature set with LOSO-CV. Purpose: a sanity reference — it tells us what "use everything" looks like before we start ablating. LR is deliberately scoped to the baseline only and is **not** part of the ablation loop.

### Step 6 — Main ablation (`ablation.py`) ← **the main-paper result**
The core pipeline. For each of the 17 subsets, it runs three classifiers:
1. **Random Forest** (co-primary)
2. **XGBoost** (co-primary)
3. **Dummy stratified** (chance floor — samples from the training class prior)

That's 17 × 3 = 51 LOSO-CV runs. Results are written to `results/ablation_metrics_{rf,xgb,dummy}.csv`, a combined `ablation_metrics.csv`, and `cross_model_ranking.csv`. The script also builds a "top-tier CI-overlap cluster" — subsets whose 95% fold-SE CI overlaps the nominal leader's CI — and flags them with a `T` in the printed summary instead of calling anything "the best."

### Step 7 — Sensitivity analyses + final reporting (`sensitivity_and_report.py`)
Re-runs RF and XGB on the **top-5 cross-model subsets + `all`** under two alternative labels (`label_p75`, `label_mage_median`). These are robustness probes, not re-ranking attempts — the script never picks a new "winner" from these runs. It then writes the final consolidated tables (`final_ablation_comparison.csv`, `final_config.json`) and prints a 7-section Step-12 interpretation that ends with the main-paper bottom line and limitations block.

---

## 4. Sensor modalities and features (what's in each subset)

| Modality | Features | n | Source on the E4 |
|---|---|---|---|
| **hr** | `hr_mean`, `hr_std`, `hr_min`, `hr_max` | 4 | PPG sensor (derived HR stream) |
| **ibi** | `ibi_mean`, `ibi_sdnn`, `ibi_rmssd`, `ibi_pnn50` | 4 | PPG sensor (inter-beat intervals / HRV) |
| **acc** | `acc_mean_mag`, `acc_std_mag`, `acc_active_pct` | 3 | Accelerometer |
| **eda** | `eda_mean`, `eda_std`, `eda_n_peaks` | 3 | Electrodermal activity |
| **temp** | `temp_mean`, `temp_std`, `temp_min` | 3 | Skin temperature |
| `hr_ibi` | HR + IBI features together | 8 | PPG (both are the same physical sensor) |

For sensor-level ablation we treat `hr_ibi` as one sensor (it *is* one sensor, physically), so the four ablation units are **{hr_ibi, acc, eda, temp}**. The 17 subsets enumerated in `feature_config.ABLATION_SUBSETS` cover every single modality, the `hr_ibi` pair, every pair of the four units, every triplet, and the full set.

---

## 5. Primary endpoint vs sensitivity endpoints

### Primary (pre-registered): `label_median`
**What it is.** A binary label that is 1 if the participant's daily CV of glucose is ≥ the cohort median CV, else 0. Roughly 50/50 split by construction.

**Why this is the primary.**
1. **Balanced.** Every classifier has enough positive and negative examples per fold to learn from. Unbalanced labels would make LOSO folds even noisier than they already are.
2. **Defensible for a normoglycemic cohort.** The AACE clinical CV cutoff (36%) was designed for diabetic populations. Our cohort is normoglycemic (mean CV ~16.7%), so almost no days exceed 36% — the clinical cutoff is effectively unusable. A cohort-relative split is the honest alternative, and we report it as such.
3. **Most stable across classifiers** in our own diagnostic runs on the `deeksha+laasya` branch — the per-fold variance and ranking are less sensitive to classifier choice than under p75.
4. **Pre-registered.** We committed to it in code (`feature_config.PRIMARY_ENDPOINT`) *before* looking at final ablation numbers, specifically so the "minimum viable sensor" finding could not be cherry-picked from whichever label happened to look best.

### Sensitivity (pre-specified robustness probes)
- **`label_mage_median`** — CV is a ratio-based variability metric; MAGE (Mean Amplitude of Glycemic Excursions) is an excursion-amplitude metric. They measure *different physiological aspects* of variability. Re-running the ablation under MAGE tells us whether the story depends on our choice of variability construct.
- **`label_p75`** — CV ≥ 75th percentile. Same CV construct as the primary, but a harder, more imbalanced (~25/75) task focused on the "most variable" days only. This probes whether the primary finding is driven by easy median-vs-median separation or actually captures something about high-variability days.

**Why sensitivity analyses matter.** A single result against a single label is easy to over-interpret. Pre-specified sensitivity runs test the *same* ablation pipeline against *different* constructions of the outcome. If the finding survives, the paper is stronger. If it doesn't, that is itself a publishable methodological observation — which is exactly what happened with MAGE (see §7).

---

## 6. Current main-results branch: `harmonized-primary-analysis`

**What this branch is.** The main-paper pipeline, scoped to the pre-registered primary endpoint only. Every script imports `PRIMARY_ENDPOINT` from `feature_config.py`, and the ablation + sensitivity scripts run against that single label. The framing, the hedged reporting language ("candidate minimal configuration" rather than "winner"), and the `REPORTING_STRATEGY.md` co-author reference all live here.

**Main findings (under the primary endpoint, daily CV ≥ cohort median):**

*Dataset.* 16 participants, 112 participant-days, 54 high / 58 low. **14 of 16 LOSO folds are evaluable** for AUROC: participants 006 (all-high) and 015 (all-low with only 2 test examples) produce single-class held-out folds where AUROC is undefined.

*Top performers (per-fold AUROC mean ± std; fold-SE 95% CI):*

| Subset | n_feat | RF | XGB | Notes |
|---|---|---|---|---|
| `acc` | 3 | **0.725 ± 0.255** [0.592, 0.859] | 0.662 ± 0.206 [0.553, 0.770] | RF's nominal best |
| `acc+temp` | 6 | 0.701 ± 0.293 [0.548, 0.855] | 0.686 ± 0.284 [0.537, 0.835] | Best cross-model mean rank |
| `acc+eda+temp` | 9 | 0.664 ± 0.274 [0.520, 0.807] | **0.693 ± 0.295** [0.538, 0.848] | XGB's nominal best |
| `all` (full E4) | 17 | 0.645 ± 0.294 [0.491, 0.798] | 0.627 ± 0.329 [0.454, 0.800] | Full-feature reference |
| *Dummy (stratified)* | — | *0.443 ± 0.198 [0.340, 0.547]* | *same* | Chance floor |

*Agreement.* Both classifiers agree on the same top-tier candidate subsets: **{`acc`, `acc+temp`, `acc+eda+temp`}** — cross-model agreement is the strongest point of confidence in the ranking.

*Dummy floor.* The stratified Dummy classifier sits at ~0.44 with upper CI 0.547. **Only RF `acc` (lower CI 0.592) cleanly clears the Dummy ceiling.** All other subsets' lower bounds approach or overlap the Dummy upper bound. This asymmetry is in the main-paper report.

**What we CAN claim.**
- In a 16-participant normoglycemic feasibility cohort, there is evidence that **a small accelerometer-dominated subset of Empatica E4 features can classify the pre-registered cohort-median CV primary endpoint above the Dummy floor, particularly for RF `acc`**, with cross-model agreement on the top candidates.
- **Adding more sensors does not consistently improve discrimination** in this cohort — the full 17-feature model is numerically matched or outperformed by 3–9-feature subsets.
- The result supports the *feasibility* of a low-burden wearable-only glycemic-variability monitor and motivates a pre-registered replication on a clinically labeled cohort.

**What we CANNOT claim.**
- No single "winning" sensor subset. Fold-SE 95% CIs overlap heavily; the top-tier cluster is statistically indistinguishable.
- No clinical classification of glycemic instability. The label is cohort-relative; the clinical AACE cutoff is not populated by this cohort.
- No generalization beyond 16 participants, this pipeline, and this label. A larger, clinically labeled replication is required before any "minimum viable sensor" claim goes beyond feasibility framing.

---

## 7. Robustness / appendix branch: `deeksha+laasya`

**Role.** Appendix / robustness source. Supplies (a) the pre-specified sensitivity-label diagnostics and the label-dependence finding, (b) the participant-stratified cluster-bootstrap methodology used for pooled AUROC CIs throughout the paper, and (c) the degenerate-fold bookkeeping (`auroc_folds_eval`). This branch is **not** a main-results branch — it was explicitly repositioned as appendix after the stability analysis.

**What it showed about label dependence.** When the same ablation pipeline is re-run against `label_mage_median` instead of the primary CV-median label, the ranking of minimal subsets **inverts**:

| Subset | RF (MAGE) | XGB (MAGE) | Compare to primary |
|---|---|---|---|
| `acc` | 0.431 ± 0.325 | 0.441 ± 0.331 | Drops from 0.725 / 0.662 |
| `acc+temp` | 0.536 ± 0.271 | 0.527 ± 0.309 | Drops from 0.701 / 0.686 |
| `acc+eda+temp` | 0.526 ± 0.299 | 0.528 ± 0.310 | Drops from 0.664 / 0.693 |
| `all` | **0.608 ± 0.274** | **0.569 ± 0.298** | Rises from 0.645 / 0.627 |

Under MAGE, the accelerometer-dominated subsets no longer lead — the full-feature model does. Importantly, **MAGE is more evenly distributed across participants** in this cohort than CV, so 15 of 16 LOSO folds are evaluable (vs 14 under the primary). The inversion is unlikely to be explained solely by fold degeneracy or poorer fold quality.

**Under `label_p75`**, performance compresses toward chance across the board (RF `acc` 0.518, RF `all` 0.371), and only 11 of 16 folds remain evaluable because the 25/75 split leaves more participants entirely on one side of the threshold.

**Why this is appendix, not main result.**
1. The appendix is precisely where "does the ranking depend on the endpoint?" belongs — it is a methodological observation about the *pipeline*, not a substantive answer to the research question.
2. Promoting MAGE to primary *after* seeing that it reorders subsets would be p-hacking. Pre-registration is what makes sensitivity meaningful, and pre-registration means picking one primary label before running.
3. Main-paper results should come from *one* clearly-scoped pipeline run. The harmonized branch does that; the appendix explains what robustness runs revealed.
4. Label-dependence is itself a publishable observation: it argues that any future minimum-sensor claim should pre-register a single variability construct, and this is exactly the kind of methodological honesty that helps a small-N feasibility paper clear review.

---

## 8. Modeling details

### Classifiers
- **Random Forest** (RF). 100 trees, `class_weight="balanced"`. Co-primary.
- **XGBoost** (XGB). 100 trees, `max_depth=3`, `learning_rate=0.1`, `scale_pos_weight` **recomputed per fold** from the training fold's class ratio. Co-primary.
- **Logistic Regression** (LR). L2, `class_weight="balanced"`, features scaled inside each fold. **Baseline only** — runs on the full 17-feature set, not inside the ablation loop.
- **Dummy stratified**. Samples from training-fold class prior. Chance floor on every ablation subset.

### Why RF and XGB are co-primary (not one or the other)
Tree ensembles are a natural fit for small tabular datasets with non-linear interactions and no strict feature-scaling requirement. RF and XGB have different inductive biases (bagging vs boosting; averaged independent trees vs sequentially corrected trees), so **agreement across both is a cheap way to check that a result is not a quirk of one model family**. We report both, and we highlight cross-model top-5 agreement as the main evidence for the candidate minimal subsets.

### Why Dummy is included
A stratified Dummy classifier on the same LOSO setup is the *only* honest "chance" comparison we have. Published AUROC numbers around 0.5 are easy to misread — showing that the Dummy actually scores ~0.44 with upper CI 0.547 on this specific dataset *because of* the participant-level variance and class distribution anchors the reader's interpretation of what "above chance" means here. It also gives reviewers a direct visual on how far each real model is from noise.

### Why LR is kept as baseline
LR is the standard linear-interpretable reference: if a linear model on all 17 features already does well, that bounds how much non-linear complexity is actually helping. But LR is excluded from the ablation loop because we do not want to mix linear and non-linear rankings, and because the study's research question is about *sensor subsets*, not about *model families*.

### How LOSO-CV works here
- **Fold = participant.** Each of the 16 participants is held out in turn; the model trains on the other 15 and predicts on the held-out participant's rows.
- **`auroc_folds_eval`** tracks how many of the 16 folds were non-degenerate (positives *and* negatives present in the held-out fold). On the primary label, that number is 14/16 because participants 006 and 015 are single-class under this label.
- **Per-fold summary.** Each fold produces one AUROC number (if evaluable); we report the mean and std across those fold AUROCs as the primary summary, with a fold-SE 95% CI (mean ± 1.96 · std / √n_folds_eval).
- **Pooled summary.** We also compute pooled out-of-fold AUROC on all rows at once, with a **participant-stratified cluster bootstrap** for the CI (resample participants with replacement, concatenate their rows, recompute AUROC per resample). This is wider than a naive row-level bootstrap but it is the only CI that respects the LOSO independence unit.

### How leakage is avoided
- **Scaling inside the fold.** For LR, `StandardScaler` is fit on the training fold and applied to the test fold — never fit on the full dataset.
- **Class-weight inside the fold.** XGBoost's `scale_pos_weight` is recomputed from the *training* fold's class ratio every fold. We never use the test fold's labels to set training weights.
- **No participant in both train and test.** LOSO's entire point: held-out participant's data never appears in training.
- **Feature extraction precedes modeling.** Feature computation happens in `cgm_processing.py` / wearable extraction, not inside the model fit, so there's no fold-dependent feature engineering to leak.
- **Hyperparameters are fixed defensible defaults, not tuned.** With N=16 participants, nested LOSO tuning is likely to produce optimistic and unstable estimates, so we committed to published defaults up front and documented them in `ablation.py`.

---

## 9. Key results to know for the presentation

### The numbers you should have memorized
- **N = 16 participants, 112 participant-days** (54 high / 58 low under the primary label).
- **14/16 folds evaluable** on the primary; **15/16 on MAGE**; **11/16 on p75**.
- **Per-fold AUROC std ≈ 0.25–0.30** across every subset — this is the dominant source of uncertainty.
- **Dummy floor**: 0.443 ± 0.198, 95% CI [0.340, 0.547].
- **RF `acc`**: 0.725 ± 0.255, [0.592, 0.859] — the only subset whose CI cleanly clears the Dummy ceiling.
- **Cross-model top-5 agreement**: `{acc, acc+temp, acc+eda+temp}`.
- **MAGE inversion**: `all` beats `acc` under MAGE (0.608 vs 0.431 for RF), exactly opposite to the primary-label ranking.

### "Top-tier CI-overlap cluster" — say it like this
"We sort subsets by per-fold AUROC mean and flag every subset whose 95% CI overlaps the nominal leader's 95% CI. Those subsets are **not statistically separable from the leader given the variance at N=16**, so we report them as a *cluster of candidate minimal configurations* rather than picking one. In this ablation the cluster contains essentially every reasonable subset — which is itself informative: it tells you the data can't resolve a single winner."

### Why we avoid naming a single winner
Because the fold-SE confidence intervals overlap. At N=16 with per-fold std ~0.25–0.30, the CI width for AUROC means is large enough that even the 17-feature full model and the 3-feature `acc`-only model are not statistically distinguishable on the primary endpoint. Naming a "winner" would misrepresent how much evidence we actually have — and would be impossible to defend in peer review.

### What happened under MAGE and p75 (one-sentence versions)
- **MAGE**: The candidate minimal subsets from the primary analysis *lose* to the full-feature model when the label is switched to MAGE-median — the ranking inverts. We treat this as label-dependence, not a contradiction.
- **p75**: Performance compresses toward chance for every subset because the positive class is smaller and more LOSO folds become single-class; we report it as a robustness probe showing the task gets harder at the tail.

---

## 10. Main limitations (keep these ready — reviewers will ask)

1. **Small sample size.** 16 participants, 112 participant-days. Per-fold AUROC has ~0.25–0.30 std. This is the dominant source of uncertainty and it drives almost every other limitation.
2. **Cohort-relative labels.** We do not use the AACE clinical CV cutoff of 36% because the cohort is normoglycemic — almost no days exceed it. Our primary label is a median split on this cohort's own CV distribution, which is feasibility-appropriate but **not a clinical classification**. A reviewer asking "is this clinical?" should get "no, explicitly not — see limitations."
3. **Degenerate LOSO folds.** Two participants (P006 all-high, P015 all-low) are single-class under the primary label, so their held-out folds produce undefined AUROC. Effective fold count is 14 of 16, surfaced in every metric table as `auroc_folds_eval`.
4. **Overlapping confidence intervals.** The 95% CIs across the top-tier cluster heavily overlap. We *cannot* separate candidate minimal subsets from each other, only from the Dummy floor — and even then only for RF `acc`.
5. **Label dependence.** The sensor-subset ranking changes when the label changes (MAGE inversion). Any future replication should pre-register a single variability construct and stick with it.
6. **Normoglycemic cohort.** Findings describe *relative* daily variability within a small healthy cohort. Generalizing to diabetic or pre-diabetic populations would require replication on those populations.
7. **Empatica E4 specifically.** The sensor set is what this device provides. A different wearable with a different sensor mix could produce different minimal subsets.

---

## 11. Presentation-ready talking points

### Five key takeaways
1. **The research question.** Can a *small* subset of wearable sensors classify daily glycemic variability in a small healthy cohort? This is a *feasibility* question, not a clinical classification question.
2. **The finding, hedged correctly.** Yes, in this cohort, an accelerometer-dominated 3-feature subset matches the full 17-feature Empatica E4 suite for classifying cohort-relative daily CV. Cross-classifier agreement strengthens the top-tier candidate list, but the 95% CIs overlap so heavily that we report a *cluster* of candidates rather than a single winner.
3. **Pre-registration is load-bearing.** We committed to the primary endpoint (`label_median`) in code before running the final ablation. MAGE and p75 are pre-specified sensitivity analyses, not alternative primaries. This is the difference between a feasibility paper and a cherry-picking exercise.
4. **Label dependence is a real finding, not a failure.** Switching from CV to MAGE inverts the subset ranking — in our report this goes into the appendix as a substantive methodological observation about how the choice of variability construct shapes wearable-subset conclusions.
5. **This is feasibility, not clinical validation.** N=16, cohort-relative labels, overlapping CIs, degenerate folds — every one of these is disclosed alongside the numbers. The paper's actual contribution is "here is an honest feasibility signal and a diagnostic showing what a larger, pre-registered replication would need."

### Five likely audience questions (with answers)

**Q1. "Why are you using a cohort-median label instead of the clinical CV 36% cutoff? Isn't 36% the accepted threshold?"**
36% is the AACE clinical cutoff for diabetic populations. Our cohort is normoglycemic with a cohort mean CV around 16.7%, so almost no days in our dataset exceed 36% — the clinical cutoff does not populate. A cohort-relative median split is the defensible alternative for a normoglycemic feasibility study, and we disclose this in the limitations and frame every finding as cohort-relative, never clinical.

**Q2. "Your confidence intervals overlap across every subset. How can you claim any of this?"**
We don't claim subset-vs-subset statistical superiority — we explicitly avoid winner language in both the code and the manuscript. What we do claim is (a) an accelerometer-dominated 3-feature subset's 95% CI cleanly clears the Dummy chance floor, which is real evidence of above-chance classification, and (b) cross-classifier top-5 agreement on the same three subsets, which is real evidence that the top cluster is not a single-model artifact. The overlapping CIs are the *reason* we report a cluster rather than a winner, and we treat the CI width itself as one of the paper's main findings about what N=16 feasibility studies can and cannot resolve.

**Q3. "Why RF and XGB as co-primary instead of picking the best one?"**
RF and XGB have different inductive biases — bagging vs boosting. If a result is real, both families should see it; if one family sees it and the other doesn't, the result is model-dependent and we should say so. Running them as co-primary and highlighting agreement on the top-5 is a cheap, honest robustness check against picking the model whose numbers happen to look best.

**Q4. "Under MAGE labeling the ranking inverts. Doesn't that contradict your main result?"**
It doesn't contradict it — it qualifies it. CV and MAGE measure different physiological aspects of variability (ratio-based dispersion vs excursion amplitude), so the features most useful for classifying "which days have higher ratio-variability" are not necessarily the ones most useful for "which days have larger excursions." That's a *methodological* observation about label choice, not a failure of the primary pipeline. We pre-registered the CV-median label as primary before seeing any of this, we report the MAGE inversion in the appendix as a label-dependence observation, and the takeaway is that any future replication should pre-register one variability construct and stick with it.

**Q5. "Why is LOSO-CV the right validation scheme here? Why not a simpler k-fold?"**
Random k-fold on participant-days would let the same participant's days appear in both training and test, which leaks *person-level* patterns into the model. Wearable features are highly person-specific (baseline HR, activity habits, skin conductance levels), so a person-leaking fold scheme would give artificially optimistic AUROC. LOSO forces the model to generalize across people, which is the question we actually care about. The cost is that some held-out participants produce single-class folds where AUROC is undefined — we track this explicitly with `auroc_folds_eval` so the effective sample size is visible next to every number.

---

## 12. Quick reference: "where is X?"

| I want to... | Look at... |
|---|---|
| See the primary endpoint definition | `feature_config.py` → `PRIMARY_ENDPOINT` |
| See what subsets are ablated | `feature_config.py` → `ABLATION_SUBSETS` |
| See how LOSO-CV is implemented | `modeling_utils.py` → `run_loso_cv()` |
| See the participant-cluster bootstrap | `modeling_utils.py` → `compute_fold_metrics()` |
| See the hyperparameter policy | `ablation.py` module docstring |
| Reproduce the main-paper numbers | `python3 ablation.py` on `harmonized-primary-analysis` |
| Reproduce the sensitivity + bottom line | `python3 sensitivity_and_report.py` |
| Read the co-author reporting decisions | `REPORTING_STRATEGY.md` |
| See the MAGE inversion numbers | `results/sensitivity_metrics_mage.csv` |
| See the p75 near-chance numbers | `results/sensitivity_metrics_p75.csv` |
| See the Dummy chance floor | `results/ablation_metrics_dummy.csv` |

---

*End of audit report. Update this document whenever the main-paper framing changes. If in doubt, prefer the code and `REPORTING_STRATEGY.md` as authoritative sources.*
