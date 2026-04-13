"""
Feature groups, sensor-ablation subsets, and pre-registered label hierarchy.

Import this module everywhere labels or feature lists are needed so that
all scripts stay in sync with a single source of truth.

Pre-registered label hierarchy (harmonized-primary-analysis branch)
-------------------------------------------------------------------
  PRIMARY     : label_median      (CV >= cohort median, ~50/50 split)
      THE single endpoint against which the main-paper sensor-ablation
      result is reported. Chosen because it is balanced, methodologically
      defensible for a normoglycemic feasibility cohort, and the most
      stable across classifiers in our diagnostics. This designation is
      pre-registered at the code level: PRIMARY_ENDPOINT below is the
      only label that may be used for main-text claims about minimum
      viable wearable sensor configurations.

  SENSITIVITY : label_p75         (CV >= 75th percentile, ~25/75 split)
      Pre-specified robustness check. Reported once in a sensitivity
      table; MUST NOT be used to re-rank or re-declare a winner.

  SENSITIVITY : label_mage_median (MAGE >= cohort median)
      Pre-specified robustness check on an alternative variability
      construct (excursion amplitude). Reported once in a sensitivity
      table; MUST NOT be used to re-rank or re-declare a winner.

  REFERENCE   : label_clinical    (CV >= 36%, AACE clinical cutoff)
      Included for traceability only. The cohort is normoglycemic, so
      this threshold is effectively unusable for classification here
      and is NOT a candidate primary or sensitivity endpoint.

Any downstream script that reports a subset ranking, a "candidate
minimum configuration," or any Dummy-floor comparison in the main
paper MUST use PRIMARY_ENDPOINT. Sensitivity scripts must explicitly
consume SENSITIVITY_LABEL_COL or MAGE_LABEL_COL and label their outputs
as sensitivity analyses.

Usage
-----
    from feature_config import MODALITIES, ABLATION_SUBSETS, ALL_FEATURES
    from feature_config import PRIMARY_ENDPOINT, LABEL_COL
    from feature_config import SENSITIVITY_LABEL_COL, MAGE_LABEL_COL
"""

from itertools import combinations

# ── Label columns ─────────────────────────────────────────────────────────────
# Pre-registered primary endpoint. `LABEL_COL` is kept as an alias for
# backward compatibility with scripts ported from the deeksha branch.
PRIMARY_ENDPOINT      = "label"             # == label_median, pre-registered primary
LABEL_COL             = PRIMARY_ENDPOINT    # legacy alias — do not repurpose
SENSITIVITY_LABEL_COL = "label_p75"         # pre-specified sensitivity (CV p75 split)
MAGE_LABEL_COL        = "label_mage_median" # pre-specified sensitivity (MAGE median split)

# ── Individual modality feature lists ─────────────────────────────────────────
MODALITIES: dict[str, list[str]] = {
    # Heart rate — from Empatica HR file (~1 Hz)
    "hr": [
        "hr_mean",   # daily mean heart rate (bpm)
        "hr_std",    # daily SD of heart rate (intra-day variability)
        "hr_min",    # daily minimum heart rate
        "hr_max",    # daily maximum heart rate
    ],

    # Inter-beat interval / HRV — from Empatica IBI file (event-driven)
    "ibi": [
        "ibi_mean",   # mean IBI (s) — inverse of average HR
        "ibi_sdnn",   # SD of all IBI values — overall autonomic variability
        "ibi_rmssd",  # root-mean-square successive differences — parasympathetic index
        "ibi_pnn50",  # % successive |ΔIBI| > 50 ms — parasympathetic index
    ],

    # Accelerometer — from Empatica ACC file (32 Hz, downsampled to 1 Hz)
    "acc": [
        "acc_mean_mag",    # daily mean vector magnitude (proxy for overall activity)
        "acc_std_mag",     # daily SD of magnitude (activity variability)
        "acc_active_pct",  # % of seconds with |mag - mean| > 15% of mean
    ],

    # Electrodermal activity — from Empatica EDA file (4 Hz)
    "eda": [
        "eda_mean",    # mean skin conductance level — tonic arousal
        "eda_std",     # SD of skin conductance — sympathetic variability
        "eda_n_peaks", # SCR event count — phasic sympathetic responses
    ],

    # Skin temperature — from Empatica TEMP file (4 Hz)
    "temp": [
        "temp_mean",  # mean skin temperature
        "temp_std",   # SD of skin temperature
        "temp_min",   # daily minimum (peripheral vasoconstriction proxy)
    ],
}

# HR and IBI both originate from the E4's PPG sensor — treated as one
# "heart" sensor for ablation purposes
MODALITIES["hr_ibi"] = MODALITIES["hr"] + MODALITIES["ibi"]

# ── Full feature list (ordered, deduplicated) ─────────────────────────────────
_seen: set[str] = set()
ALL_FEATURES: list[str] = []
for _feats in MODALITIES.values():
    for _f in _feats:
        if _f not in _seen:
            _seen.add(_f)
            ALL_FEATURES.append(_f)

# ── Sensor-ablation subsets ───────────────────────────────────────────────────
# The four "sensor-level" units for combinatorial ablation:
#   hr_ibi  (PPG-derived, 8 features)
#   acc     (accelerometer, 3 features)
#   eda     (EDA, 3 features)
#   temp    (temperature, 3 features)
_ABLATION_UNITS = ["hr_ibi", "acc", "eda", "temp"]


def _merge(*keys: str) -> list[str]:
    """Return deduplicated feature list for the given modality keys."""
    seen: set[str] = set()
    out: list[str] = []
    for k in keys:
        for f in MODALITIES[k]:
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


ABLATION_SUBSETS: dict[str, list[str]] = {}

# Singles — raw modalities (5 subsets)
for _m in ["hr", "ibi", "acc", "eda", "temp"]:
    ABLATION_SUBSETS[_m] = MODALITIES[_m]

# HR+IBI as one natural unit (1 subset)
ABLATION_SUBSETS["hr_ibi"] = MODALITIES["hr_ibi"]

# All pairs of ablation units (6 subsets)
for _pair in combinations(_ABLATION_UNITS, 2):
    ABLATION_SUBSETS["+".join(_pair)] = _merge(*_pair)

# All triplets of ablation units (4 subsets)
for _trip in combinations(_ABLATION_UNITS, 3):
    ABLATION_SUBSETS["+".join(_trip)] = _merge(*_trip)

# Full set (1 subset)
ABLATION_SUBSETS["all"] = ALL_FEATURES

# ── Metadata ──────────────────────────────────────────────────────────────────
# Columns that are NOT features (CGM metrics, labels, metadata)
META_COLS = {
    "participant", "date", "n_readings",
    "mean_glucose", "sd_glucose", "cv",
    "tir_70_180", "tar_180", "tbr_70", "mage",
    "label", "label_str",
    "label_median", "label_p75", "label_clinical", "label_mage_median",
    "cv_threshold_median", "cv_threshold_p75", "mage_threshold_median",
}

# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total features : {len(ALL_FEATURES)}")
    print(f"Ablation subsets: {len(ABLATION_SUBSETS)}\n")
    print(f"{'Subset':<22s}  {'n':>3s}  Features")
    print("-" * 70)
    for name, feats in ABLATION_SUBSETS.items():
        n = len(feats)
        if name in ("hr", "ibi", "acc", "eda", "temp"):
            tag = "[single]"
        elif name == "hr_ibi":
            tag = "[paired]"
        elif name.count("+") == 1:
            tag = "[pair]  "
        elif name.count("+") == 2:
            tag = "[triplet]"
        else:
            tag = "[full]  "
        print(f"  {name:<20s}  {n:>3d}  {tag}  {feats}")
    print(f"\nPrimary label   : {LABEL_COL}")
    print(f"Sensitivity label: {SENSITIVITY_LABEL_COL}")
