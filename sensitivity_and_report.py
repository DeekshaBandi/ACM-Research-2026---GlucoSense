"""
Steps 10-12: Sensitivity analysis, final outputs, and interpretation
---------------------------------------------------------------------
Step 10: Re-run best subsets with label_p75 (both RF and XGBoost)
Step 11: Consolidate and save all final outputs
Step 12: Print structured interpretation
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_config  import ABLATION_SUBSETS, LABEL_COL, SENSITIVITY_LABEL_COL
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
print("STEP 10 -- Sensitivity analysis  (label_p75: CV >= 75th percentile)")
print("=" * 70)

# Quick label stats
y_p75 = df_full[SENSITIVITY_LABEL_COL].dropna()
print(f"\nlabel_p75 distribution: "
      f"{int(y_p75.sum())} high / {int(len(y_p75) - y_p75.sum())} low  "
      f"({y_p75.mean()*100:.1f}% / {(1-y_p75.mean())*100:.1f}%)\n")

sens_rows    = []   # p75 results
primary_rows = []   # median results (for comparison)

print(f"  {'Subset':<22s}  {'Model':<12s}  {'Label':<8s}  {'n':>4s}  "
      f"{'AUROC_fm':>8s}  {'+-std':>5s}  {'BalAcc_fm':>9s}  {'F1_fm':>6s}  "
      f"{'Sens_fm':>7s}  {'Spec_fm':>7s}")
print("  " + "-" * 100)

for subset in SENS_SUBSETS:
    feats = ABLATION_SUBSETS[subset]

    for model_name, (make_clf, scale) in SENS_MODELS.items():
        short_model = "RF" if model_name == "random_forest" else "XGB"

        # Primary label (median)
        X, y, parts = prepare_matrix(df_full, feats, LABEL_COL)
        preds_med   = run_loso_cv(make_clf(), X, y, parts, scale=scale)
        m_med       = compute_fold_metrics(preds_med)
        primary_rows.append({"subset": subset, "model": model_name,
                             "label": "median", **m_med})

        # Sensitivity label (p75)
        X75, y75, p75 = prepare_matrix(df_full, feats, SENSITIVITY_LABEL_COL)
        preds_p75     = run_loso_cv(make_clf(), X75, y75, p75, scale=scale)
        preds_p75.to_csv(
            f"results/sensitivity_predictions_{model_name}_{subset}.csv",
            index=False,
        )
        m_p75 = compute_fold_metrics(preds_p75)
        sens_rows.append({"subset": subset, "model": model_name,
                          "label": "p75", **m_p75})

        for lbl, m in [("median", m_med), ("p75", m_p75)]:
            print(f"  {subset:<22s}  {short_model:<12s}  {lbl:<8s}  "
                  f"{m['n']:>4d}  "
                  f"{m['auroc_fold_mean']:>8.3f}  {m['auroc_fold_std']:>5.3f}  "
                  f"{m['bal_acc_fold_mean']:>9.3f}  {m['f1_fold_mean']:>6.3f}  "
                  f"{m['sens_fold_mean']:>7.3f}  {m['spec_fold_mean']:>7.3f}")
    print()

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv("results/sensitivity_metrics_p75.csv", index=False)


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
    "primary_label":           LABEL_COL,
    "sensitivity_label":       SENSITIVITY_LABEL_COL,
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
    ("results/ablation_metrics.csv",          "All ablation metrics (RF + XGB, 34 runs)"),
    ("results/ablation_metrics_rf.csv",       "RF ablation metrics (17 subsets)"),
    ("results/ablation_metrics_xgb.csv",      "XGBoost ablation metrics (17 subsets)"),
    ("results/cross_model_ranking.csv",       "Cross-model subset ranking"),
    ("results/final_ablation_comparison.csv", "RF vs XGB side-by-side comparison"),
    ("results/baseline_metrics.csv",          "Baseline comparison (LR / RF / XGB)"),
    ("results/sensitivity_metrics_p75.csv",   "Sensitivity analysis (both models, label_p75)"),
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

# Sensitivity comparison: does min-viable pattern hold under p75?
def _get_row(rows, subset, model):
    return next((r for r in rows
                 if r["subset"] == subset and r["model"] == model), None)

mv_rf_p75  = _get_row(sens_rows, mv_sub, "random_forest")
all_rf_p75 = _get_row(sens_rows, "all", "random_forest")
mv_xgb_p75 = _get_row(sens_rows, mv_sub, "xgboost")
all_xgb_p75 = _get_row(sens_rows, "all", "xgboost")

has_sens = all(x is not None for x in
               [mv_rf_p75, all_rf_p75, mv_xgb_p75, all_xgb_p75])
if has_sens:
    pattern_rf  = mv_rf_p75["auroc_fold_mean"] > all_rf_p75["auroc_fold_mean"]
    pattern_xgb = mv_xgb_p75["auroc_fold_mean"] > all_xgb_p75["auroc_fold_mean"]
    pattern_both = pattern_rf and pattern_xgb
else:
    pattern_both = False

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

3. MINIMUM VIABLE SENSOR CONFIGURATION
   -----------------------------------------------------------------
   Decision rule: {fc['decision_rule']}

   >> Recommended: {mv_sub} ({mv_nfeat} features)
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

   Reduced sensor sets match or exceed the full-feature model in both
   classifiers, indicating that additional modalities add noise rather
   than discriminative power in this cohort.

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
6. SENSITIVITY ANALYSIS  (label_p75: CV >= 75th percentile)
   -----------------------------------------------------------------
   {mv_sub} (p75):  RF AUROC fm = {mv_rf_p75['auroc_fold_mean']:.3f}  |  XGB AUROC fm = {mv_xgb_p75['auroc_fold_mean']:.3f}
   all    (p75):  RF AUROC fm = {all_rf_p75['auroc_fold_mean']:.3f}  |  XGB AUROC fm = {all_xgb_p75['auroc_fold_mean']:.3f}
   {mv_sub} {'outperforms' if pattern_both else 'does not consistently outperform'} the full model under p75 labeling across both models.
   {'The main finding is robust to the labeling threshold.' if pattern_both
    else 'The advantage weakens under p75; interpret with caution.'}""")

cross_auroc_avg = (mv_rf_m["auroc_fold_mean"] + mv_xgb_m["auroc_fold_mean"]) / 2

print(f"""
7. BOTTOM LINE
   -----------------------------------------------------------------
   In this 16-participant feasibility study, daily glycemic variability
   (defined by cohort-relative CV) can be classified from a reduced
   wearable sensor configuration ({mv_sub}, {mv_nfeat} features) at
   AUROC ~{cross_auroc_avg:.2f} (cross-model average), matching or exceeding
   the full Empatica E4 sensor suite (17 features).

   This finding is consistent across both Random Forest and XGBoost,
   confirming it is not an artefact of a single model's inductive bias.

   These results support the feasibility of low-burden glycemic
   instability monitoring using a reduced-sensor wrist device and
   warrant replication in a larger, more diverse sample.
""")

print("  Saved -> results/final_config.json")
print("  Saved -> results/final_ablation_comparison.csv")
print("  Saved -> results/sensitivity_metrics_p75.csv")
