import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import warnings

# Import the logic from feature_engineering
from feature_engineering import engineer_features

warnings.filterwarnings("ignore")

def load_and_sync_dataset():
    """
    Orchestrates the data pipeline:
    1. Finds participant folders
    2. Loads labels (Ground Truth)
    3. Calls external feature engineering
    4. Aligns sensors to labels temporally
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, 'data', 'glycemic_data')
    
    all_data_frames = []
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    print(f"Starting pipeline for {len(folders)} participants...")
    
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

            # 2. Call the new modular feature engineering
            features_dict = engineer_features(p_path, p_id)
            if not features_dict:
                continue

            # 3. Temporal Alignment (Merge sensor features onto Dexcom timestamps)
            merged_df = gdf.copy()
            for feat_name, series in features_dict.items():
                merged_df = pd.merge_asof(
                    merged_df.sort_index(), 
                    series.to_frame(feat_name).sort_index(), 
                    left_index=True, 
                    right_index=True,
                    direction='backward'  # Fix 3: direction='backward' ensures only past sensor data is used, preventing future leakage
                )

            # Clean up missing data (where sensors might have been off)
            clean_df = merged_df.dropna(subset=[col for col in merged_df.columns if '_mean_5' in col])
            
            if not clean_df.empty:
                clean_df['participant_id'] = p_id
                all_data_frames.append(clean_df)
                print(f"  [SUCCESS] ID {p_id}: Synced {len(clean_df)} windows")
            
        except Exception as e:
            print(f"  [ERROR] ID {p_id}: {e}")

    return pd.concat(all_data_frames) if all_data_frames else pd.DataFrame()

def main():
    # --- DATA INGESTION ---
    df = load_and_sync_dataset()
    if df.empty:
        print("No data found. Check file paths and label files.")
        return

    # --- CLASS BALANCING ---
    # Fix 4: Replace downsampling with class weights
    # Remove downsampling to keep all data, use class weights instead
    print(f"\nTraining on Full Dataset: {len(df)} samples.")

    # --- PREPARATION ---
    # Identify all engineered columns (excluding labels and rolling stats)
    feature_cols = [c for c in df.columns if any(suffix in c for suffix in ['_5', '_30', '_10', '_ratio'])]
    
    # Fix 5: Add temporal (time-based) train/test split as the PRIMARY split
    # Combine with Fix 1: group by participant AND respect temporal ordering
    # Temporal split per participant to avoid participant leakage and respect time ordering
    train_dfs = []
    test_dfs = []
    for p_id, group in df.groupby('participant_id'):
        group = group.sort_index()  # Sort by Timestamp index
        cutoff = int(len(group) * 0.8)
        train_dfs.append(group.iloc[:cutoff])
        test_dfs.append(group.iloc[cutoff:])
    
    df_train = pd.concat(train_dfs)
    df_test = pd.concat(test_dfs)
    
    X_train = df_train[feature_cols]
    y_train = le.fit_transform(df_train['Pers_Label'])
    X_test = df_test[feature_cols]
    y_test = le.transform(df_test['Pers_Label'])
    
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # --- MODEL TRAINING ---
    # Standard XGBoost config from the Bent et al. methodology
    model = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        random_state=42,
        objective='multi:softprob'
    )
    model.fit(X_train, y_train, sample_weight=weights)

    # --- EVALUATION ---
    y_pred = model.predict(X_test)
    
    print("\n" + "="*60)
    print("XGBOOST CLASSIFICATION REPORT (Modular Workflow)")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"FINAL MACRO F1 SCORE: {macro_f1:.4f}")

    # --- FEATURE ANALYSIS ---
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTOP 10 CONTRIBUTING FEATURES:")
    print(importance.head(10))

if __name__ == "__main__":
    main()


# -------------- OUTPUT --------------
# Training on Balanced Dataset: 5171 samples per class (15513 total).

# ============================================================
# XGBOOST CLASSIFICATION REPORT (Modular Workflow)
# ============================================================
#               precision    recall  f1-score   support

#     PersHigh       0.58      0.64      0.61      1034
#      PersLow       0.58      0.65      0.62      1034
#     PersNorm       0.59      0.45      0.51      1035

#     accuracy                           0.58      3103
#    macro avg       0.58      0.58      0.58      3103
# weighted avg       0.58      0.58      0.58      3103

# FINAL MACRO F1 SCORE: 0.5779

# TOP 10 CONTRIBUTING FEATURES:
# hr_mean_30      0.117146
# eda_mean_5      0.056701
# eda_mean_30     0.054357
# ibi_mean_30     0.052684
# hr_acc_ratio    0.045854
# temp_mean_30    0.045704
# acc_mean_30     0.044712
# temp_mean_5     0.044459
# acc_std_5       0.040481
# bvp_std_5       0.038507
# dtype: float32