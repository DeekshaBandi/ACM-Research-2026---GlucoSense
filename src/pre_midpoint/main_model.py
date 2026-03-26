import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import warnings

# Import our updated modular engineering logic
from feature_engineering import engineer_features

warnings.filterwarnings("ignore")

def load_and_sync_dataset():
    """
    Pipeline: Loads data per participant, engineers features, 
    aligns timestamps, and applies Subject-Level Normalization.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, 'data', 'glycemic_data')
    
    all_data_frames = []
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    print(f"Starting Ultimate Pipeline: Normalization + Lags...")
    
    for p_id in folders:
        p_path = os.path.join(data_dir, p_id)
        label_file = os.path.join(p_path, f"Dexcom_{p_id}_labeled.csv")
        
        if not os.path.exists(label_file):
            continue
            
        try:
            # 1. Load Ground Truth Labels
            gdf = pd.read_csv(label_file)
            gdf['Timestamp'] = pd.to_datetime(gdf['Timestamp']).dt.floor('s')
            gdf = gdf.sort_values('Timestamp').groupby('Timestamp').first()

            # 2. Call modular feature engineering (includes new Lags)
            features_dict = engineer_features(p_path, p_id)
            if not features_dict:
                continue

            # 3. Temporal Alignment
            merged_df = gdf.copy()
            for feat_name, series in features_dict.items():
                merged_df = pd.merge_asof(
                    merged_df.sort_index(), 
                    series.to_frame(feat_name).sort_index(), 
                    left_index=True, 
                    right_index=True
                )

            # Drop rows missing core sensor data
            clean_df = merged_df.dropna(subset=[col for col in merged_df.columns if '_mean_5' in col])
            
            # --- SUBJECT-LEVEL NORMALIZATION (Z-Score) ---
            # This levels the playing field between different participants' baselines
            feat_cols = [c for c in clean_df.columns if any(s in c for s in ['_5', '_30', '_10', '_ratio', '_lag'])]
            
            for col in feat_cols:
                m_val = clean_df[col].mean()
                s_val = clean_df[col].std()
                if s_val > 0:
                    clean_df[col] = (clean_df[col] - m_val) / s_val
            
            if not clean_df.empty:
                all_data_frames.append(clean_df)
                print(f"  [SUCCESS] ID {p_id}: Synced, Lagged & Normalized")
            
        except Exception as e:
            print(f"  [ERROR] ID {p_id}: {e}")

    return pd.concat(all_data_frames) if all_data_frames else pd.DataFrame()

def main():
    # 1. DATA INGESTION
    df = load_and_sync_dataset()
    if df.empty:
        print("No data found. Check your file paths.")
        return

    # 2. CLASS BALANCING
    # Balance based on the smallest class to avoid 'Normal' bias
    counts = df['Pers_Label'].value_counts()
    min_size = counts.min()
    
    df_bal = pd.concat([
        resample(df[df['Pers_Label'] == label], replace=False, n_samples=min_size, random_state=42)
        for label in df['Pers_Label'].unique()
    ])
    
    print(f"\nTraining on Balanced Dataset: {min_size} samples per class.")

    # 3. PREPARATION
    feature_cols = [c for c in df_bal.columns if any(s in c for s in ['_5', '_30', '_10', '_ratio', '_lag'])]
    X = df_bal[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(df_bal['Pers_Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. ADVANCED XGBOOST CONFIGURATION
    # Increased estimators and deeper trees to capture subtle personalized patterns
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective='multi:softprob',
        tree_method='hist'
    )
    
    model.fit(X_train, y_train)

    # 5. EVALUATION
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("ULTIMATE PASSIVE GLUCOSE MODEL: FINAL RESULTS")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    m_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"FINAL MACRO F1 SCORE: {m_f1:.4f}")

    # 6. FEATURE IMPORTANCE
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTOP 15 CONTRIBUTING FEATURES (Normalized & Lagged):")
    print(importance.head(15))

if __name__ == "__main__":
    main()


# ============================================================
# ULTIMATE PASSIVE GLUCOSE MODEL: FINAL RESULTS
# ============================================================
#               precision    recall  f1-score   support

#     PersHigh       0.61      0.65      0.63      1034
#      PersLow       0.62      0.70      0.66      1034
#     PersNorm       0.61      0.49      0.54      1035

#     accuracy                           0.61      3103
#    macro avg       0.61      0.61      0.61      3103
# weighted avg       0.61      0.61      0.61      3103

# FINAL MACRO F1 SCORE: 0.6105

# TOP 15 CONTRIBUTING FEATURES (Normalized & Lagged):
# hr_mean_30            0.057984
# eda_mean_30           0.045062
# eda_mean_5_lag_15     0.042204
# temp_mean_5_lag_15    0.041909
# hr_acc_ratio          0.040684
# temp_mean_30          0.040675
# ibi_mean_30           0.039602
# hr_mean_5_lag_15      0.037735
# eda_mean_5            0.037099
# eda_std_5             0.035226
# ibi_slope_10          0.034626
# acc_mean_30           0.034480
# temp_mean_5           0.034165
# bvp_std_5             0.034087
# acc_std_5             0.033494
# dtype: float32

# [Done] exited with code=0 in 785.374 seconds