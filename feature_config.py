"""
Feature groups and sensor-ablation subsets.

Import this module everywhere labels or feature lists are needed so that
all scripts stay in sync with a single source of truth.

Label strategy
--------------
  PRIMARY   : label_median  (CV >= cohort median, ~50/50 split)
  SENSITIVITY: label_p75    (CV >= 75th percentile, ~25/75 split)
  REFERENCE : label_clinical (CV >= 36%, effectively unusable for this cohort)

Usage
-----
    from feature_config import MODALITIES, ABLATION_SUBSETS, ALL_FEATURES
    from feature_config import LABEL_COL, SENSITIVITY_LABEL_COL
"""

from itertools import combinations

# ── Label columns ─────────────────────────────────────────────────────────────
LABEL_COL             = "label"          # == label_median, used for all primary models
SENSITIVITY_LABEL_COL = "label_p75"      # used in sensitivity analysis only

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
    "label_median", "label_p75", "label_clinical",
    "cv_threshold_median", "cv_threshold_p75",
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
