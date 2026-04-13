"""
Sensor Ablation Study (Co-Primary: RF + XGBoost) + Dummy floor
---------------------------------------------------------------
Evaluates Random Forest and XGBoost as co-primary models across all 17
feature subsets defined in feature_config.ABLATION_SUBSETS.

A stratified DummyClassifier is also run on every subset to establish an
explicit chance floor (sanity check — reviewers should be able to see at
a glance how far each real model is above random guessing).

Logistic Regression is deliberately kept BASELINE-ONLY (see train_baseline.py)
and is NOT part of the ablation loop: RF and XGB are the co-primary models,
LR provides a linear-interpretable reference on the full feature set only.

Goal: identify wearable sensor configurations that perform consistently well
across both non-trivial models. "Best" subsets are reported with per-fold
uncertainty, not ranked by point estimates alone.

Hyperparameter policy
---------------------
Model hyperparameters are FIXED defensible defaults — not tuned on this
dataset. Tuning inside LOSO with 16 participants is likely to produce
optimistic and unstable estimates. The chosen defaults are:

  RandomForest: n_estimators=100, class_weight="balanced"
    Rationale: sklearn default tree count, balanced weighting to handle
    the ~50/50 cohort-median label without per-fold leakage.

  XGBoost: n_estimators=100, max_depth=3, learning_rate=0.1
    Rationale: shallow trees + standard learning rate are a conservative
    default for small N (~112 rows); scale_pos_weight is recomputed per
    LOSO fold inside run_loso_cv to avoid leaking test-fold class ratios.

  Dummy: strategy="stratified" — samples from training class prior.

Subsets:
  Singles  : hr, ibi, acc, eda, temp
  Paired   : hr_ibi
  Pairs    : hr_ibi+acc, hr_ibi+eda, hr_ibi+temp, acc+eda, acc+temp, eda+temp
  Triplets : hr_ibi+acc+eda, hr_ibi+acc+temp, hr_ibi+eda+temp, acc+eda+temp
  Full     : all

Outputs:
  results/ablation_predictions_{model}_{subset}.csv  — per-row predictions
  results/ablation_metrics.csv                       — full metric table (RF + XGB)
  results/ablation_metrics_rf.csv                    — RF metrics
  results/ablation_metrics_xgb.csv                   — XGBoost metrics
  results/cross_model_ranking.csv                    — cross-model subset ranking
  results/ensemble_metrics.csv                       — soft-voting ensemble (secondary)
  results/model_selection.json                       — co-primary rationale + min viable config
"""

import os, json, warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from feature_config import ABLATION_SUBSETS, LABEL_COL
from modeling_utils  import load_data, prepare_matrix, run_loso_cv, compute_fold_metrics

warnings.filterwarnings("ignore")
os.makedirs("results", exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────
# See module docstring for hyperparameter justification (fixed defaults — no
# per-fold tuning). XGBoost scale_pos_weight is recomputed inside run_loso_cv.

MODELS = {
    "dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=1,  # overridden per-fold by run_loso_cv
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    ),
}
SCALE = {"dummy_stratified": False, "random_forest": False, "xgboost": False}


# ── Ablation loop ─────────────────────────────────────────────────────────────

all_rows = []
ensemble_rows = []

n_subsets = len(ABLATION_SUBSETS)
n_models  = len(MODELS)
total     = n_subsets * n_models

print(f"Running ablation: {n_subsets} subsets x {n_models} models = {total} runs\n")
print(f"{'#':>3s}  {'Subset':<22s}  {'Model':<16s}  "
      f"{'n':>4s}  {'fe':>3s}  {'AUROC_fm':>8s}  {'+-std':>5s}  "
      f"{'95% CI (fold SE)':>18s}  {'BalAcc':>6s}  {'F1':>6s}")
print("-" * 110)

run_i = 0
for subset_name, feat_cols in ABLATION_SUBSETS.items():
    df_data = load_data()
    X, y, parts = prepare_matrix(df_data, feat_cols, LABEL_COL)

    subset_preds = {}
    for model_name, clf in MODELS.items():
        run_i += 1
        scale = SCALE[model_name]

        preds = run_loso_cv(clf, X, y, parts, scale=scale)
        preds.to_csv(
            f"results/ablation_predictions_{model_name}_{subset_name}.csv",
            index=False,
        )

        m = compute_fold_metrics(preds)

        row = {
            "subset":     subset_name,
            "n_features": len(feat_cols),
            "model":      model_name,
            **m,
        }
        all_rows.append(row)
        subset_preds[model_name] = preds

        ci_str = f"[{m['auroc_fold_ci_lo']:.2f}, {m['auroc_fold_ci_hi']:.2f}]"
        print(f"{run_i:>3d}  {subset_name:<22s}  {model_name:<16s}  "
              f"{m['n']:>4d}  {m['auroc_folds_eval']:>3d}  "
              f"{m['auroc_fold_mean']:>8.3f}  {m['auroc_fold_std']:>5.3f}  "
              f"{ci_str:>18s}  "
              f"{m['bal_acc_fold_mean']:>6.3f}  {m['f1_fold_mean']:>6.3f}")

    # ── Soft-voting ensemble (RF + XGB only; dummy excluded) ─────────────
    rf_p  = subset_preds["random_forest"]
    xgb_p = subset_preds["xgboost"]
    assert np.array_equal(rf_p["y_true"].values, xgb_p["y_true"].values), \
        f"Prediction alignment error for subset {subset_name}"
    ens = rf_p[["participant", "fold", "y_true"]].copy()
    ens["y_prob"] = (rf_p["y_prob"].values + xgb_p["y_prob"].values) / 2
    ens["y_pred"] = (ens["y_prob"] >= 0.5).astype(int)
    ens.to_csv(
        f"results/ablation_predictions_ensemble_{subset_name}.csv", index=False,
    )
    m_ens = compute_fold_metrics(ens)
    ensemble_rows.append({
        "subset": subset_name, "n_features": len(feat_cols),
        "model": "ensemble", **m_ens,
    })


# ── Save per-model results ────────────────────────────────────────────────────
results = pd.DataFrame(all_rows)
results.to_csv("results/ablation_metrics.csv", index=False)

dummy_results = results[results["model"] == "dummy_stratified"].copy()
rf_results    = results[results["model"] == "random_forest"].copy()
xgb_results   = results[results["model"] == "xgboost"].copy()
dummy_results.to_csv("results/ablation_metrics_dummy.csv", index=False)
rf_results.to_csv("results/ablation_metrics_rf.csv", index=False)
xgb_results.to_csv("results/ablation_metrics_xgb.csv", index=False)

ens_df = pd.DataFrame(ensemble_rows)
ens_df.to_csv("results/ensemble_metrics.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Per-model summaries
# ══════════════════════════════════════════════════════════════════════════════

def print_model_summary(model_df, top_set=None):
    """
    Ranked subset table for one model.

    top_set: set of subset names in a "top tier" (overlapping-CI cluster).
             Rows in this set are flagged with 'T' instead of a hard '<< BEST'
             to avoid implying statistical separation when CIs heavily overlap.
    """
    s = model_df.sort_values("auroc_fold_mean", ascending=False)
    print(f"\n  {'Rank':<4s} {'Subset':<22s} {'nf':>3s} {'n':>4s} {'fe':>3s}  "
          f"{'AUROC_fm':>8s}  {'+-std':>5s}  {'95% CI (fold SE)':>18s}  "
          f"{'BalAcc':>6s}  {'F1':>6s}  {'Sens':>5s}  {'Spec':>5s}")
    print("  " + "-" * 110)
    for rank, (_, r) in enumerate(s.iterrows(), 1):
        flags = []
        if top_set is not None and r["subset"] in top_set:
            flags.append("T")          # inside top-tier (CI-overlap cluster)
        if r["subset"] == "all":
            flags.append("FULL")
        tag = ("  " + " ".join(flags)) if flags else ""
        ci_str = f"[{r.auroc_fold_ci_lo:.2f}, {r.auroc_fold_ci_hi:.2f}]"
        print(f"  {rank:<4d} {r['subset']:<22s} "
              f"{int(r.n_features):>3d} {int(r.n):>4d} {int(r.auroc_folds_eval):>3d}  "
              f"{r.auroc_fold_mean:>8.3f}  {r.auroc_fold_std:>5.3f}  "
              f"{ci_str:>18s}  "
              f"{r.bal_acc_fold_mean:>6.3f}  {r.f1_fold_mean:>6.3f}  "
              f"{r.sens_fold_mean:>5.3f}  {r.spec_fold_mean:>5.3f}{tag}")


def _top_tier(model_df):
    """
    Return the set of subsets whose 95% per-fold-mean CI overlaps the
    top subset's CI. These are NOT statistically distinguishable from the
    nominal leader given LOSO variance.
    """
    s = model_df.sort_values("auroc_fold_mean", ascending=False)
    top = s.iloc[0]
    top_lo = top["auroc_fold_ci_lo"]
    overlap = s[s["auroc_fold_ci_hi"] >= top_lo]
    return set(overlap["subset"].tolist())


rf_top  = _top_tier(rf_results)
xgb_top = _top_tier(xgb_results)

print("\n")
print("=" * 110)
print("  DUMMY (stratified) floor -- per-subset chance baseline")
print("  All non-dummy rows should be interpreted RELATIVE to these numbers.")
print("=" * 110)
print_model_summary(dummy_results)

print("\n")
print("=" * 110)
print("  ABLATION SUMMARY -- Random Forest, ranked by AUROC fold mean")
print("  (T = in top-tier cluster whose 95% CI overlaps the nominal leader's CI;")
print("   these subsets are NOT statistically separable from the leader)")
print("=" * 110)
print_model_summary(rf_results, top_set=rf_top)

print("\n")
print("=" * 110)
print("  ABLATION SUMMARY -- XGBoost, ranked by AUROC fold mean")
print("  (T = top-tier cluster, see note above)")
print("=" * 110)
print_model_summary(xgb_results, top_set=xgb_top)


# ══════════════════════════════════════════════════════════════════════════════
# Cross-model comparison
# ══════════════════════════════════════════════════════════════════════════════

RANK_METRICS = [
    "auroc_fold_mean", "bal_acc_fold_mean", "f1_fold_mean",
    "sens_fold_mean", "spec_fold_mean",
]

rf_idx  = rf_results.set_index("subset")
xgb_idx = xgb_results.set_index("subset")

# Pre-compute per-metric ranks for each model
rf_ranks  = {m: rf_idx[m].rank(ascending=False)  for m in RANK_METRICS}
xgb_ranks = {m: xgb_idx[m].rank(ascending=False) for m in RANK_METRICS}

cross_rows = []
for subset in rf_idx.index:
    row = {
        "subset":     subset,
        "n_features": int(rf_idx.loc[subset, "n_features"]),
    }
    all_ranks_list = []
    for m in RANK_METRICS:
        short = m.replace("_fold_mean", "")
        row[f"rf_{short}"]       = rf_idx.loc[subset, m]
        row[f"xgb_{short}"]      = xgb_idx.loc[subset, m]
        row[f"rf_rank_{short}"]  = rf_ranks[m][subset]
        row[f"xgb_rank_{short}"] = xgb_ranks[m][subset]
        all_ranks_list.extend([rf_ranks[m][subset], xgb_ranks[m][subset]])

    row["mean_rank"]        = np.mean(all_ranks_list)
    row["cross_auroc_mean"] = (rf_idx.loc[subset, "auroc_fold_mean"]
                               + xgb_idx.loc[subset, "auroc_fold_mean"]) / 2
    row["cross_auroc_min"]  = min(rf_idx.loc[subset, "auroc_fold_mean"],
                                  xgb_idx.loc[subset, "auroc_fold_mean"])
    cross_rows.append(row)

cross_df = (
    pd.DataFrame(cross_rows)
    .sort_values("mean_rank")
    .reset_index(drop=True)
)
cross_df.to_csv("results/cross_model_ranking.csv", index=False)

# Highlight cross-model agreement in top-5
rf_top5  = set(rf_results.sort_values("auroc_fold_mean", ascending=False)
               .head(5)["subset"])
xgb_top5 = set(xgb_results.sort_values("auroc_fold_mean", ascending=False)
               .head(5)["subset"])
agree_top5 = rf_top5 & xgb_top5

print("\n\n")
print("=" * 80)
print("  CROSS-MODEL COMPARISON -- ranked by mean rank (5 metrics x 2 models)")
print("=" * 80)
print(f"\n  {'Rank':<5s}  {'Subset':<22s}  {'n_feat':>6s}  "
      f"{'RF_AUC':>7s}  {'XGB_AUC':>7s}  {'Avg_AUC':>7s}  {'Min_AUC':>7s}  "
      f"{'MeanRnk':>7s}")
print("  " + "-" * 80)
for rank, (_, r) in enumerate(cross_df.iterrows(), 1):
    tag = " *" if r["subset"] in agree_top5 else ""
    print(f"  {rank:<5d}  {r['subset']:<22s}  {int(r['n_features']):>6d}  "
          f"{r['rf_auroc']:>7.3f}  {r['xgb_auroc']:>7.3f}  "
          f"{r['cross_auroc_mean']:>7.3f}  {r['cross_auroc_min']:>7.3f}  "
          f"{r['mean_rank']:>7.2f}{tag}")

print(f"\n  * = in both models' top 5 by AUROC: {sorted(agree_top5)}")


# ══════════════════════════════════════════════════════════════════════════════
# Minimum viable sensor configuration
# ══════════════════════════════════════════════════════════════════════════════

MARGIN = 0.05

best_cross_min = cross_df["cross_auroc_min"].max()
viable = cross_df[cross_df["cross_auroc_min"] >= best_cross_min - MARGIN]
min_viable = viable.sort_values("n_features").iloc[0]

best_rf  = rf_results.sort_values("auroc_fold_mean", ascending=False).iloc[0]
best_xgb = xgb_results.sort_values("auroc_fold_mean", ascending=False).iloc[0]

print(f"\n\n  == CANDIDATE MINIMAL SENSOR CONFIGURATIONS ==")
print(f"\n  NOTE: Per-fold AUROC std on this dataset is ~0.25-0.30, so top")
print(f"        subsets overlap substantially in uncertainty. The 'nominal")
print(f"        minimum viable' pick below is a point-estimate heuristic,")
print(f"        NOT a claim of statistical superiority. Reviewers should")
print(f"        treat all 'T'-flagged subsets as plausible candidates.")
print(f"\n  Decision rule (nominal only):")
print(f"    Smallest subset where BOTH models' AUROC fold mean >= "
      f"{best_cross_min - MARGIN:.3f}")
print(f"    (= best cross-model min {best_cross_min:.3f} minus "
      f"{MARGIN} tolerance)")
print(f"\n  Per-model nominal best (point estimate only):")
print(f"    RF  : {best_rf['subset']:<16s}  AUROC fm = "
      f"{best_rf['auroc_fold_mean']:.3f} +- {best_rf['auroc_fold_std']:.3f}")
print(f"    XGB : {best_xgb['subset']:<16s}  AUROC fm = "
      f"{best_xgb['auroc_fold_mean']:.3f} +- {best_xgb['auroc_fold_std']:.3f}")
print(f"\n  >> Nominal minimum-viable subset: {min_viable['subset']}")
print(f"    Features       : {int(min_viable['n_features'])}")
print(f"    RF AUROC fm    : {min_viable['rf_auroc']:.3f}")
print(f"    XGB AUROC fm   : {min_viable['xgb_auroc']:.3f}")
print(f"    Cross-model avg: {min_viable['cross_auroc_mean']:.3f}")
print(f"    Mean rank      : {min_viable['mean_rank']:.2f}")

print(f"\n  All viable subsets (within {MARGIN} of best, by size):")
for _, v in viable.sort_values("n_features").iterrows():
    print(f"    {v['subset']:<22s}  ({int(v['n_features']):>2d} feat)  "
          f"RF={v['rf_auroc']:.3f}  XGB={v['xgb_auroc']:.3f}  "
          f"rank={v['mean_rank']:.2f}")


# ── Optional: ensemble summary ────────────────────────────────────────────────

print("\n\n")
print("=" * 80)
print("  OPTIONAL -- Soft-voting ensemble (RF + XGB), ranked by AUROC fold mean")
print("=" * 80)
print_model_summary(ens_df)


# ── Save model selection ─────────────────────────────────────────────────────

rf_mean_rank = cross_df[
    [c for c in cross_df.columns if c.startswith("rf_rank_")]
].mean().mean()
xgb_mean_rank = cross_df[
    [c for c in cross_df.columns if c.startswith("xgb_rank_")]
].mean().mean()

model_sel = {
    "approach": "co-primary",
    "models": ["random_forest", "xgboost"],
    "rationale": (
        f"RF and XGBoost evaluated as co-primary models across 17 sensor "
        f"subsets. Overall mean rank (5 metrics): RF {rf_mean_rank:.2f}, "
        f"XGB {xgb_mean_rank:.2f}. Neither model consistently dominates; "
        f"both are retained to confirm findings are not model-dependent."
    ),
    "best_rf_subset":  best_rf["subset"],
    "best_rf_auroc_fm": round(float(best_rf["auroc_fold_mean"]), 3),
    "best_xgb_subset": best_xgb["subset"],
    "best_xgb_auroc_fm": round(float(best_xgb["auroc_fold_mean"]), 3),
    "minimum_viable_subset": min_viable["subset"],
    "minimum_viable_n_features": int(min_viable["n_features"]),
    "minimum_viable_rf_auroc": round(float(min_viable["rf_auroc"]), 3),
    "minimum_viable_xgb_auroc": round(float(min_viable["xgb_auroc"]), 3),
    "decision_rule": (
        f"Smallest subset where both models' AUROC fold mean >= "
        f"{best_cross_min:.3f} - {MARGIN} = {best_cross_min - MARGIN:.3f}"
    ),
}
with open("results/model_selection.json", "w") as f:
    json.dump(model_sel, f, indent=2)


print(f"\n\n  Saved -> results/ablation_metrics.csv")
print(f"  Saved -> results/ablation_metrics_rf.csv")
print(f"  Saved -> results/ablation_metrics_xgb.csv")
print(f"  Saved -> results/cross_model_ranking.csv")
print(f"  Saved -> results/ensemble_metrics.csv")
print(f"  Saved -> results/model_selection.json")
