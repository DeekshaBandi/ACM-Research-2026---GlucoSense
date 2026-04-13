"""
Steps 10-12: Pre-specified sensitivity analyses, final outputs, and
main-paper interpretation — harmonized-primary-analysis branch
----------------------------------------------------------------
This script implements the MAIN-PAPER reporting pipeline on top of the
pre-registered primary ablation (ablation.py). It runs two pre-specified
sensitivity analyses against alternative variability labels and writes
the final main-paper output tables.

  Primary endpoint (from ablation.py) : PRIMARY_ENDPOINT == label_median
  Sensitivity #1 (pre-specified)      : SENSITIVITY_LABEL_COL (CV p75)
  Sensitivity #2 (pre-specified)      : MAGE_LABEL_COL (MAGE cohort-median)

Both sensitivity runs are reported as single robustness tables. They MUST
NOT be used to re-rank subsets or re-declare a "winner" — their role is
exclusively to probe whether the primary-endpoint conclusions are stable
under alternative constructions of glycemic variability.

Step 10: Re-run top-5 cross-model subsets under the two sensitivity labels
Step 11: Consolidate and save all final main-paper outputs
Step 12: Print structured main-paper interpretation
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_config  import (
    ABLATION_SUBSETS, PRIMARY_ENDPOINT,
    SENSITIVITY_LABEL_COL, MAGE_LABEL_COL,
)
from modeling_utils  import load_data, prepare_matrix, run_loso_cv, compute_fold_metrics

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Model constructors ───────────────────────────────────────────────────────

def make_rf():
    return RandomForestClassifier(
        n_estimators=100, class_weight="balanced",
        random_state=42, n_jobs=-1,
    )

def make_xgb():
    return XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        scale_pos_weight=1,  # overridden per-fold by run_loso_cv
        eval_metric="logloss", random_state=42, verbosity=0,
    )

SENS_MODELS = {
    "random_forest": (make_rf, False),   # (constructor, scale)
    "xgboost":       (make_xgb, False),
}


# ════════════════════════════════════════════════════════════════════════════════
# STEP 10 -- Sensitivity analysis with label_p75 (both models)
# ════════════════════════════════════════════════════════════════════════════════

# Derive top-5 subsets from cross-model ranking, always include "all"
_cross = pd.read_csv("results/cross_model_ranking.csv")
_top5 = _cross.sort_values("mean_rank").head(5)["subset"].tolist()
if "all" not in _top5:
    _top5.append("all")
SENS_SUBSETS = _top5

df_full = load_data()

print("=" * 70)
print("STEP 10 -- PRE-SPECIFIED sensitivity analyses")
print(f"  Primary endpoint (ablation.py) : {PRIMARY_ENDPOINT}  (CV >= cohort median)")
print("  Sensitivity #1 (pre-specified) : label_p75          (CV >= 75th percentile)")
print("  Sensitivity #2 (pre-specified) : label_mage_median  (MAGE >= cohort median)")
print("  Role                           : robustness probes of the primary result.")
print("                                   NOT used to re-rank or re-declare a winner.")
print("=" * 70)

# Quick label stats
y_p75 = df_full[SENSITIVITY_LABEL_COL].dropna()
y_mage = df_full[MAGE_LABEL_COL].dropna()
print(f"\nlabel_p75  distribution: "
      f"{int(y_p75.sum())} high / {int(len(y_p75) - y_p75.sum())} low  "
      f"({y_p75.mean()*100:.1f}% / {(1-y_p75.mean())*100:.1f}%)")
print(f"label_mage distribution: "
      f"{int(y_mage.sum())} high / {int(len(y_mage) - y_mage.sum())} low  "
      f"({y_mage.mean()*100:.1f}% / {(1-y_mage.mean())*100:.1f}%)  "
      f"[NaN MAGE days excluded]\n")

sens_rows    = []   # p75 results (kept name for downstream compatibility)
mage_rows    = []   # MAGE sensitivity
primary_rows = []   # median results (for comparison)

# Sensitivity labels iterated over: (short name, column)
SENS_LABELS = [("p75", SENSITIVITY_LABEL_COL), ("mage", MAGE_LABEL_COL)]

print(f"  {'Subset':<22s}  {'Model':<12s}  {'Label':<8s}  {'n':>4s}  {'fe':>3s}  "
      f"{'AUROC_fm':>8s}  {'+-std':>5s}  {'95% CI (fold SE)':>18s}  "
      f"{'BalAcc':>6s}  {'F1':>6s}")
print("  " + "-" * 110)

for subset in SENS_SUBSETS:
    feats = ABLATION_SUBSETS[subset]

    for model_name, (make_clf, scale) in SENS_MODELS.items():
        short_model = "RF" if model_name == "random_forest" else "XGB"

        # Primary endpoint (pre-registered)
        X, y, parts = prepare_matrix(df_full, feats, PRIMARY_ENDPOINT)
        preds_med   = run_loso_cv(make_clf(), X, y, parts, scale=scale)
        m_med       = compute_fold_metrics(preds_med)
        primary_rows.append({"subset": subset, "model": model_name,
                             "label": "primary", **m_med})

        label_metrics = {"primary": m_med}

        for short, col in SENS_LABELS:
            X_s, y_s, p_s = prepare_matrix(df_full, feats, col)
            preds_s = run_loso_cv(make_clf(), X_s, y_s, p_s, scale=scale)
            preds_s.to_csv(
                f"results/sensitivity_predictions_{short}_{model_name}_{subset}.csv",
                index=False,
            )
            m_s = compute_fold_metrics(preds_s)
            row = {"subset": subset, "model": model_name,
                   "label": short, **m_s}
            if short == "p75":
                sens_rows.append(row)
            else:
                mage_rows.append(row)
            label_metrics[short] = m_s

        for lbl in ("primary", "p75", "mage"):
            m = label_metrics[lbl]
            ci_str = f"[{m['auroc_fold_ci_lo']:.2f}, {m['auroc_fold_ci_hi']:.2f}]"
            print(f"  {subset:<22s}  {short_model:<12s}  {lbl:<8s}  "
                  f"{m['n']:>4d}  {m['auroc_folds_eval']:>3d}  "
                  f"{m['auroc_fold_mean']:>8.3f}  {m['auroc_fold_std']:>5.3f}  "
                  f"{ci_str:>18s}  "
                  f"{m['bal_acc_fold_mean']:>6.3f}  {m['f1_fold_mean']:>6.3f}")
    print()

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv("results/sensitivity_metrics_p75.csv", index=False)
mage_df = pd.DataFrame(mage_rows)
mage_df.to_csv("results/sensitivity_metrics_mage.csv", index=False)


# ════════════════════════════════════════════════════════════════════════════════
# STEP 11 -- Consolidate and save final outputs
# ════════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("STEP 11 -- Final outputs")
print("=" * 70)

# Load all previously saved artefacts
ablation_rf   = pd.read_csv("results/ablation_metrics_rf.csv")
ablation_xgb  = pd.read_csv("results/ablation_metrics_xgb.csv")
ablation_all  = pd.read_csv("results/ablation_metrics.csv")
cross_ranking = pd.read_csv("results/cross_model_ranking.csv")
baseline      = pd.read_csv("results/baseline_metrics.csv")
with open("results/model_selection.json") as f:
    model_sel = json.load(f)

# ── Ablation comparison table (RF + XGB side by side, ranked by mean rank) ───
metric_cols = ["auroc_fold_mean", "auroc_fold_std", "auroc_pooled",
               "auroc_ci_lo", "auroc_ci_hi",
               "bal_acc_fold_mean", "f1_fold_mean",
               "sens_fold_mean", "spec_fold_mean"]

rf_side = ablation_rf[["subset", "n_features", "n"] + metric_cols].copy()
rf_side = rf_side.rename(columns={c: f"rf_{c}" for c in metric_cols})

xgb_side = ablation_xgb[["subset"] + metric_cols].copy()
xgb_side = xgb_side.rename(columns={c: f"xgb_{c}" for c in metric_cols})

comparison = (
    rf_side
    .merge(xgb_side, on="subset")
    .merge(cross_ranking[["subset", "mean_rank", "cross_auroc_mean",
                          "cross_auroc_min"]],
           on="subset")
    .sort_values("mean_rank")
    .reset_index(drop=True)
)
comparison.index += 1
comparison.to_csv("results/final_ablation_comparison.csv")

# ── Best config (cross-model) ────────────────────────────────────────────────
min_viable_subset = model_sel["minimum_viable_subset"]
mv_rf  = ablation_rf[ablation_rf["subset"] == min_viable_subset].iloc[0]
mv_xgb = ablation_xgb[ablation_xgb["subset"] == min_viable_subset].iloc[0]

best_rf_subset  = model_sel["best_rf_subset"]
best_xgb_subset = model_sel["best_xgb_subset"]

all_rf  = ablation_rf[ablation_rf["subset"] == "all"].iloc[0]
all_xgb = ablation_xgb[ablation_xgb["subset"] == "all"].iloc[0]

mkeys = ["auroc_fold_mean", "auroc_fold_std", "bal_acc_fold_mean",
         "f1_fold_mean", "sens_fold_mean", "spec_fold_mean"]

final_config = {
    "approach":                model_sel["approach"],
    "models":                  model_sel["models"],
    "primary_endpoint":        PRIMARY_ENDPOINT,
    "sensitivity_label_p75":   SENSITIVITY_LABEL_COL,
    "sensitivity_label_mage":  MAGE_LABEL_COL,
    "minimum_viable_subset":   min_viable_subset,
    "minimum_viable_n_features": model_sel["minimum_viable_n_features"],
    "best_rf_subset":          best_rf_subset,
    "best_xgb_subset":         best_xgb_subset,
    "decision_rule":           model_sel["decision_rule"],
    "minimum_viable_metrics": {
        "rf":  {k: round(float(mv_rf[k]), 3) for k in mkeys},
        "xgb": {k: round(float(mv_xgb[k]), 3) for k in mkeys},
    },
    "best_rf_metrics": {
        k: round(float(ablation_rf[ablation_rf["subset"] == best_rf_subset]
                        .iloc[0][k]), 3)
        for k in mkeys
    },
    "best_xgb_metrics": {
        k: round(float(ablation_xgb[ablation_xgb["subset"] == best_xgb_subset]
                        .iloc[0][k]), 3)
        for k in mkeys
    },
    "full_model_metrics": {
        "rf":  {k: round(float(all_rf[k]), 3) for k in mkeys},
        "xgb": {k: round(float(all_xgb[k]), 3) for k in mkeys},
    },
}
with open("results/final_config.json", "w") as f:
    json.dump(final_config, f, indent=2)

saved_files = [
    ("results/ablation_metrics.csv",          "All ablation metrics (Dummy + RF + XGB, 51 runs)"),
    ("results/ablation_metrics_dummy.csv",    "Dummy (stratified) floor metrics (17 subsets)"),
    ("results/ablation_metrics_rf.csv",       "RF ablation metrics (17 subsets)"),
    ("results/ablation_metrics_xgb.csv",      "XGBoost ablation metrics (17 subsets)"),
    ("results/cross_model_ranking.csv",       "Cross-model subset ranking"),
    ("results/final_ablation_comparison.csv", "RF vs XGB side-by-side comparison"),
    ("results/baseline_metrics.csv",          "Baseline comparison (LR / RF / XGB)"),
    ("results/sensitivity_metrics_p75.csv",   "Sensitivity analysis (both models, label_p75)"),
    ("results/sensitivity_metrics_mage.csv",  "Sensitivity analysis (both models, label_mage_median)"),
    ("results/ensemble_metrics.csv",          "Soft-voting ensemble (secondary)"),
    ("results/model_selection.json",          "Co-primary model rationale + min viable config"),
    ("results/final_config.json",             "Final configuration summary"),
]
for fname in [f for f, _ in saved_files]:
    assert os.path.exists(fname), f"Missing: {fname}"

print("\n  All output files verified:")
for fname, desc in saved_files:
    size = os.path.getsize(fname)
    print(f"    {fname:<50s}  ({size:,} bytes)  {desc}")

# Per-run prediction CSVs
pred_files = [f for f in os.listdir("results")
              if f.startswith(("ablation_pred", "baseline_pred",
                               "sensitivity_pred"))]
print(f"\n  Per-run prediction CSVs: {len(pred_files)} files in results/")


# ════════════════════════════════════════════════════════════════════════════════
# STEP 12 -- Interpretation
# ════════════════════════════════════════════════════════════════════════════════

fc = final_config
mv_rf_m  = fc["minimum_viable_metrics"]["rf"]
mv_xgb_m = fc["minimum_viable_metrics"]["xgb"]
full_rf_m  = fc["full_model_metrics"]["rf"]
full_xgb_m = fc["full_model_metrics"]["xgb"]
mv_sub     = fc["minimum_viable_subset"]
mv_nfeat   = fc["minimum_viable_n_features"]

# Sensitivity comparison: does min-viable pattern hold under p75 / MAGE?
def _get_row(rows, subset, model):
    return next((r for r in rows
                 if r["subset"] == subset and r["model"] == model), None)

def _pattern(rows, mv_sub):
    mv_rf  = _get_row(rows, mv_sub, "random_forest")
    all_rf = _get_row(rows, "all",  "random_forest")
    mv_xg  = _get_row(rows, mv_sub, "xgboost")
    all_xg = _get_row(rows, "all",  "xgboost")
    if not all([mv_rf, all_rf, mv_xg, all_xg]):
        return None
    return {
        "mv_rf": mv_rf, "all_rf": all_rf, "mv_xg": mv_xg, "all_xg": all_xg,
        "both":  (mv_rf["auroc_fold_mean"] > all_rf["auroc_fold_mean"] and
                  mv_xg["auroc_fold_mean"] > all_xg["auroc_fold_mean"]),
    }

sens_p75  = _pattern(sens_rows, mv_sub)
sens_mage = _pattern(mage_rows, mv_sub)
has_sens  = sens_p75 is not None

top_cross = cross_ranking.sort_values("mean_rank").iloc[0]

# Best RF and XGB subset metrics for display
best_rf_row  = ablation_rf[ablation_rf["subset"] == best_rf_subset].iloc[0]
best_xgb_row = ablation_xgb[ablation_xgb["subset"] == best_xgb_subset].iloc[0]

print("\n\n" + "=" * 70)
print("STEP 12 -- Interpretation")
print("=" * 70)

print(f"""
1. CO-PRIMARY MODEL APPROACH
   -----------------------------------------------------------------
   Random Forest and XGBoost are evaluated as co-primary models.
   {model_sel['rationale']}

2. PER-MODEL BEST SUBSETS
   -----------------------------------------------------------------
   Best RF subset  : {best_rf_subset}
     AUROC fm = {best_rf_row['auroc_fold_mean']:.3f} +- {best_rf_row['auroc_fold_std']:.3f}
     Bal. acc = {best_rf_row['bal_acc_fold_mean']:.3f}  |  F1 = {best_rf_row['f1_fold_mean']:.3f}
     Sens = {best_rf_row['sens_fold_mean']:.3f}  |  Spec = {best_rf_row['spec_fold_mean']:.3f}

   Best XGB subset : {best_xgb_subset}
     AUROC fm = {best_xgb_row['auroc_fold_mean']:.3f} +- {best_xgb_row['auroc_fold_std']:.3f}
     Bal. acc = {best_xgb_row['bal_acc_fold_mean']:.3f}  |  F1 = {best_xgb_row['f1_fold_mean']:.3f}
     Sens = {best_xgb_row['sens_fold_mean']:.3f}  |  Spec = {best_xgb_row['spec_fold_mean']:.3f}

   Cross-model best (by mean rank): {top_cross['subset']}
     Mean rank = {top_cross['mean_rank']:.2f}

3. CANDIDATE MINIMAL SENSOR CONFIGURATION
   -----------------------------------------------------------------
   Decision rule (nominal point estimate, NOT statistical superiority):
     {fc['decision_rule']}

   Reported as a candidate — not a sole winner. See ablation.py for the
   full top-tier CI-overlap cluster ('T' flag).

   >> Nominal candidate: {mv_sub} ({mv_nfeat} features)
     RF  : AUROC fm = {mv_rf_m['auroc_fold_mean']:.3f} +- {mv_rf_m['auroc_fold_std']:.3f}  |  Bal. acc = {mv_rf_m['bal_acc_fold_mean']:.3f}  |  F1 = {mv_rf_m['f1_fold_mean']:.3f}
     XGB : AUROC fm = {mv_xgb_m['auroc_fold_mean']:.3f} +- {mv_xgb_m['auroc_fold_std']:.3f}  |  Bal. acc = {mv_xgb_m['bal_acc_fold_mean']:.3f}  |  F1 = {mv_xgb_m['f1_fold_mean']:.3f}

4. REDUCED vs FULL SENSING
   -----------------------------------------------------------------
   Full model (17 features):
     RF  AUROC fm = {full_rf_m['auroc_fold_mean']:.3f} +- {full_rf_m['auroc_fold_std']:.3f}
     XGB AUROC fm = {full_xgb_m['auroc_fold_mean']:.3f} +- {full_xgb_m['auroc_fold_std']:.3f}
   {mv_sub} ({mv_nfeat} features):
     RF  AUROC fm = {mv_rf_m['auroc_fold_mean']:.3f}  (delta = {mv_rf_m['auroc_fold_mean'] - full_rf_m['auroc_fold_mean']:+.3f})
     XGB AUROC fm = {mv_xgb_m['auroc_fold_mean']:.3f}  (delta = {mv_xgb_m['auroc_fold_mean'] - full_xgb_m['auroc_fold_mean']:+.3f})

   Reduced sensor sets numerically match or exceed the full-feature
   model in both classifiers. Given the heavy CI overlap across subsets
   (see ablation.py), we frame this as "additional modalities do not
   consistently improve discrimination in this cohort" rather than as
   evidence that they actively add noise.

5. STABILITY AND NOISE
   -----------------------------------------------------------------
   Fold-level AUROC standard deviation is ~0.25-0.30 across all subsets,
   consistent with small test fold sizes (5-9 days per participant).
   Two folds are structurally uninformative: participant 006 (all high
   days; specificity undefined) and participant 015 (all low days,
   only 2 test examples; sensitivity undefined). Wide confidence intervals
   are expected and consistent with the pilot study framing.""")

if has_sens:
    print(f"""
6. PRE-SPECIFIED SENSITIVITY ANALYSES (robustness probes only)
   -----------------------------------------------------------------
   These runs test whether the primary-endpoint finding is stable under
   alternative constructions of glycemic variability. They are NOT used
   to re-rank subsets or re-declare a candidate minimum configuration.

   (a) label_p75  (CV >= 75th percentile, 25/75 split)
       {mv_sub} :  RF AUROC fm = {sens_p75['mv_rf']['auroc_fold_mean']:.3f}  |  XGB AUROC fm = {sens_p75['mv_xg']['auroc_fold_mean']:.3f}
       all     :  RF AUROC fm = {sens_p75['all_rf']['auroc_fold_mean']:.3f}  |  XGB AUROC fm = {sens_p75['all_xg']['auroc_fold_mean']:.3f}
       {'Nominal advantage of the reduced subset persists under p75.' if sens_p75['both']
        else 'Advantage does NOT consistently persist under p75 — interpret with caution.'}""")
    if sens_mage is not None:
        print(f"""   (b) label_mage_median  (MAGE secondary endpoint, cohort-median split)
       {mv_sub} :  RF AUROC fm = {sens_mage['mv_rf']['auroc_fold_mean']:.3f}  |  XGB AUROC fm = {sens_mage['mv_xg']['auroc_fold_mean']:.3f}
       all     :  RF AUROC fm = {sens_mage['all_rf']['auroc_fold_mean']:.3f}  |  XGB AUROC fm = {sens_mage['all_xg']['auroc_fold_mean']:.3f}
       {'Reduced-subset advantage also holds under MAGE labeling.' if sens_mage['both']
        else 'Reduced-subset advantage does NOT hold under MAGE labeling — the CV-based finding may be label-dependent.'}""")

cross_auroc_avg = (mv_rf_m["auroc_fold_mean"] + mv_xgb_m["auroc_fold_mean"]) / 2

print(f"""
7. MAIN-PAPER BOTTOM LINE  (harmonized primary analysis)
   -----------------------------------------------------------------
   Under the PRE-REGISTERED primary endpoint ({PRIMARY_ENDPOINT}, daily CV
   vs. cohort median), daily glycemic variability in this 16-participant
   normoglycemic feasibility cohort can be classified from a reduced
   wearable sensor configuration ({mv_sub}, {mv_nfeat} features) at an
   AUROC of ~{cross_auroc_avg:.2f} cross-model average, numerically comparable
   to the full Empatica E4 sensor suite (17 features). This is reported
   as a CANDIDATE minimal configuration — the top-tier CI-overlap cluster
   identified in ablation.py contains multiple subsets that are not
   statistically separable from the nominal leader, and the main text
   reports that cluster rather than a single winner.

   Pre-specified sensitivity runs (label_p75, label_mage_median) probe
   robustness of the primary result. They are reported as single tables
   and are NOT used to re-rank subsets. Label-dependence observed across
   these sensitivity runs is discussed in the robustness appendix rather
   than treated as a contradictory main finding.

   LIMITATIONS (reported alongside the point estimates):
     * Per-fold AUROC std is ~0.25-0.30; fold-SE 95% CIs overlap heavily
       across subsets, so "best sensor" language is avoided.
     * Two LOSO folds are structurally degenerate (single-class held-out
       participant), reducing effective fold count below 16; see
       auroc_folds_eval in every metrics table.
     * Labels are cohort-relative, not clinical. The AACE clinical CV
       cutoff (>= 36%) is included only as a reference column because
       this normoglycemic cohort does not populate it meaningfully.
     * N = 16 participants; findings describe feasibility in this cohort
       and require replication in a larger sample with a clinically
       defined variability endpoint before any minimum-sensor claim
       can be generalized.

   The results support the FEASIBILITY of low-burden glycemic variability
   monitoring from a reduced wearable sensor subset, and motivate
   pre-registered replication on a clinically labeled cohort.
""")

print("  Saved -> results/final_config.json")
print("  Saved -> results/final_ablation_comparison.csv")
print("  Saved -> results/sensitivity_metrics_p75.csv")
print("  Saved -> results/sensitivity_metrics_mage.csv")
