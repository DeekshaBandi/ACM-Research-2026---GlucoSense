import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler
import os


def run_personalized_model(filename):
    # 1. Load and Prepare
    df = pd.read_csv(filename)
    patient_id = os.path.basename(filename).split('.')[0]

    # Logic to identify unique 24-hour periods
    df['day_night'] = (df['hour'].diff() < 0).cumsum()
    # Logic to identify sleep vs awake
    df['sleep'] = df['hour'].apply(lambda x: 1 if x >= 23 or x <= 7 else 0)
    # Helper for filtering (must be dropped later)
    df['hypo_duration'] = df['hypoglycemia'].fillna(0).groupby((df['hypoglycemia'].fillna(0) == 0).cumsum()).cumcount()

    # 2. Select columns (Same as create_day_dataset but adding our helper)
    features = [
        'hr_10min_rolling', 'hr_30min_rolling', 'hr_60min_rolling',
        'last_measured_hrv_15min', 'hrv_change_15min',
        'step_count_rollingsum_30min', 'step_count_rollingsum_60min',
        'active_energy_rollingsum_30min', 'active_energy_rollingsum_60min',
        'time_since_lastmeal_ord', 'IOB', 'sin_hour', 'cos_hour'
    ]
    target = 'hypoglycemia'
    helpers = ['sleep', 'day_night', 'hypo_duration']

    df_ml = df[features + [target] + helpers].copy()
    df_ml = df_ml[(df_ml[target].notna()) & (df_ml['sleep'] == 0)]

    hypo_days = df_ml[df_ml[target] == 1]['day_night'].unique()

    all_results = []

    # 3. Modeling Loop (Leave-One-Day-Out)
    for test_id in hypo_days:
        train = df_ml[df_ml.day_night != test_id]
        train = train[train.hypo_duration <= 180]  # Filter long events

        test = df_ml[df_ml.day_night == test_id]
        test = test[test.hypo_duration <= 180]

        X_train = train[features]  # <--- FIX: ONLY USE SENSOR FEATURES
        y_train = train[target]
        X_test = test[features]
        y_test = test[target]

        # Handle class imbalance
        sampler = RandomUnderSampler(sampling_strategy=1)
        X_res, y_res = sampler.fit_resample(X_train, y_train)

        # Scale and Train
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_res)
        X_test_scaled = scaler.transform(X_test)

        model = xgb.XGBClassifier(reg_alpha=10, reg_lambda=10, eval_metric='logloss')
        model.fit(X_train_scaled, y_res)

        # Evaluate
        y_scores = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_scores)

        all_results.append({
            'Patient': patient_id,
            'Day_ID': test_id,
            'AUC': auc(fpr, tpr)
        })

    return pd.DataFrame(all_results)


# --- RUN FOR ALL FILES ---
data_files = ['../DATA/01.csv', '../DATA/02.csv', '../DATA/03.csv', '../DATA/04.csv', '../DATA/10.csv']
final_report = []

for f in data_files:
    print(f"Processing Patient {f}...")
    try:
        res = run_personalized_model(f)
        final_report.append(res)
    except Exception as e:
        print(f"Skipping {f}: {e}")

summary = pd.concat(final_report)
print("\n--- FINAL RESEARCH RESULTS ---")
print(summary.groupby('Patient')['AUC'].mean())
print(f"\nOverall Study Mean AUROC: {summary['AUC'].mean():.3f}")