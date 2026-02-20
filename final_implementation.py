import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import recall_score, roc_curve, auc, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler


def run_personalized_pipeline(file_list):
    results_storage = []
    importance_accumulator = []
    # Global master list of potential features
    master_features = [
        'hr_10min_rolling', 'hr_30min_rolling', 'hr_60min_rolling',
        'last_measured_hrv_15min', 'hrv_change_15min',
        'step_count_rollingsum_30min', 'step_count_rollingsum_60min',
        'active_energy_rollingsum_30min', 'active_energy_rollingsum_60min',
        'time_since_lastmeal_ord', 'IOB', 'sin_hour', 'cos_hour'
    ]

    for path in file_list:
        p_id = os.path.basename(path).split('.')[0]

        data = pd.read_csv(path)
        data['day_night'] = (data['hour'].diff() < 0).cumsum()
        data['sleep'] = data['hour'].apply(lambda x: 1 if x >= 23 or x <= 7 else 0)
        data['hypo_duration'] = data['hypoglycemia'].fillna(0).groupby(
            (data['hypoglycemia'].fillna(0) == 0).cumsum()).cumcount()

        # Filter features based on availability in the current file
        current_features = [f for f in master_features if f in data.columns]

        subset = data[current_features + ['hypoglycemia', 'sleep', 'day_night', 'hypo_duration']].copy()
        subset = subset[(subset['hypoglycemia'].notna()) & (subset['sleep'] == 0)]

        target_days = subset[subset['hypoglycemia'] == 1]['day_night'].unique()

        for day in target_days:
            train_set = subset[subset.day_night != day]
            train_set = train_set[train_set.hypo_duration <= 180]

            test_set = subset[subset.day_night == day]
            test_set = test_set[test_set.hypo_duration <= 180]

            # Use dynamic feature list for training and testing
            X_train, y_train = train_set[current_features], train_set['hypoglycemia']
            X_test, y_test = test_set[current_features], test_set['hypoglycemia']

            rus = RandomUnderSampler(sampling_strategy=1, random_state=42)
            X_res, y_res = rus.fit_resample(X_train, y_train)

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_res)
            X_test_s = scaler.transform(X_test)

            clf = xgb.XGBClassifier(reg_alpha=1, reg_lambda=1, eval_metric='logloss')
            clf.fit(X_train_s, y_res)

            y_pred = clf.predict(X_test_s)
            y_prob = clf.predict_proba(X_test_s)[:, 1]

            sens = recall_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            area = auc(fpr, tpr)

            results_storage.append({'Patient': p_id, 'AUC': area, 'Sensitivity': sens, 'Specificity': spec})

            # Map importance back to master feature list for consistent plotting
            fold_importance = np.zeros(len(master_features))
            for idx, feat in enumerate(current_features):
                master_idx = master_features.index(feat)
                fold_importance[master_idx] = clf.feature_importances_[idx]
            importance_accumulator.append(fold_importance)

    summary_df = pd.DataFrame(results_storage)
    patient_report = summary_df.groupby('Patient').mean()

    avg_feature_imp = np.mean(importance_accumulator, axis=0)
    plt.figure(figsize=(10, 6))
    plt.barh(master_features, avg_feature_imp, color='teal')
    plt.xlabel("Gain (Predictive Power)")
    plt.title("XGBoost Global Feature Importance")
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    return patient_report


search_path = '../DATA/'
csv_files = sorted(glob.glob(os.path.join(search_path, '*.csv')))
final_metrics = run_personalized_pipeline(csv_files)

print("\n" + "=" * 45)
print("ALGORITHM PERFORMANCE SUMMARY")
print("=" * 45)
print(final_metrics)

print("\n" + "-" * 45)
print("OVERALL STUDY MEANS")
print("-" * 45)
print(f"Mean AUROC:       {final_metrics['AUC'].mean():.3f}")
print(f"Mean Sensitivity: {final_metrics['Sensitivity'].mean():.3f}")
print(f"Mean Specificity: {final_metrics['Specificity'].mean():.3f}")
print("=" * 45)