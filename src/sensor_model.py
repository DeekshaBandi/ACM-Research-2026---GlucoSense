import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings("ignore")

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, 'data', 'glycemic_data')
    
    all_data_frames = []
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    folders.sort()
    
    print(f"Syncing the 6 Passive Wearable Streams (Optimized) for {len(folders)} participants...")
    
    for p_id in folders:
        p_path = os.path.join(data_dir, p_id)
        label_file = os.path.join(p_path, f"Dexcom_{p_id}_labeled.csv")
        if not os.path.exists(label_file): continue
            
        try:
            # 1. Load Labels
            gdf = pd.read_csv(label_file)
            gdf['Timestamp'] = pd.to_datetime(gdf['Timestamp']).dt.floor('s')
            gdf = gdf.sort_values('Timestamp')

            # 2. Optimized Sensor Loading
            def load_sensor_fast(name):
                f_path = os.path.join(p_path, f"{name}_{p_id}.csv")
                if not os.path.exists(f_path): return pd.DataFrame()
                
                df = pd.read_csv(f_path)
                df.columns = [c.strip() for c in df.columns]
                
                # Floor to second and mean-aggregate to reduce join rows by 32x-64x
                df['datetime'] = pd.to_datetime(df['datetime']).dt.floor('s')
                df = df.groupby('datetime').mean() 
                return df.sort_index()

            hr = load_sensor_fast("HR")
            temp = load_sensor_fast("TEMP")
            eda = load_sensor_fast("EDA")
            bvp = load_sensor_fast("BVP")
            acc = load_sensor_fast("ACC")
            ibi = load_sensor_fast("IBI")

            # 3. Feature Engineering (5-minute rolling windows)
            features_dict = {}
            if not hr.empty: features_dict['hr'] = hr['hr'].rolling('5min').mean()
            if not temp.empty: features_dict['temp'] = temp['temp'].rolling('5min').mean()
            if not eda.empty: features_dict['eda'] = eda['eda'].rolling('5min').mean()
            if not bvp.empty: features_dict['bvp'] = bvp['bvp'].rolling('5min').std()
            if not ibi.empty: features_dict['ibi'] = ibi['ibi'].rolling('5min').mean()
            if not acc.empty:
                # Calculating Magnitude from the 1-sec averaged X, Y, Z
                acc['mag'] = np.sqrt(acc['acc_x']**2 + acc['acc_y']**2 + acc['acc_z']**2)
                features_dict['acc'] = acc['mag'].rolling('5min').mean()

            # 4. Temporal Alignment (Now lightning fast due to reduced rows)
            m = gdf.copy()
            for name, series in features_dict.items():
                m = pd.merge_asof(m, series.to_frame(name), left_on='Timestamp', right_index=True)

            clean_m = m.dropna(subset=['hr', 'acc', 'temp', 'eda'])
            if not clean_m.empty:
                all_data_frames.append(clean_m)
                print(f"ID {p_id}: Synced {len(clean_m)} windows")
            
        except Exception as e:
            print(f"ID {p_id}: Error - {e}")

    return pd.concat(all_data_frames)

def main():
    df = load_data()
    
    # Strictly the 6 wearable features + personal baseline logic via XGBoost
    features = ['hr', 'temp', 'eda', 'bvp', 'acc', 'ibi']
    
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['Pers_Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print(f"\nTraining Passive Sensor-Only XGBoost Model...")
    # Using slightly deeper trees to compensate for missing HbA1c context
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    
    print("\n" + "="*50)
    print("STRATEGY 1: PASSIVE SENSOR-ONLY RESULTS")
    print("="*50)
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))
    
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- FEATURE IMPORTANCE ---")
    print(importance)

if __name__ == "__main__":
    main()




# ------ OUTPUT ------
# Syncing the 6 Passive Wearable Streams (Optimized) for 16 participants...
# ID 001: Synced 2561 windows
# ID 002: Synced 2119 windows
# ID 003: Synced 2214 windows
# ID 004: Synced 2164 windows
# ID 005: Synced 2557 windows
# ID 006: Synced 2847 windows
# ID 007: Synced 2207 windows
# ID 008: Synced 2505 windows
# ID 009: Synced 2306 windows
# ID 010: Synced 2148 windows
# ID 011: Synced 2843 windows
# ID 012: Synced 2169 windows
# ID 013: Synced 1979 windows
# ID 014: Synced 1878 windows
# ID 015: Synced 1673 windows
# ID 016: Synced 2277 windows

# Training Passive Sensor-Only XGBoost Model...

# ==================================================
# STRATEGY 1: PASSIVE SENSOR-ONLY RESULTS
# ==================================================
#               precision    recall  f1-score   support

#     PersHigh       0.27      0.56      0.36      1117
#      PersLow       0.27      0.64      0.38      1034
#     PersNorm       0.85      0.41      0.56      5139

#     accuracy                           0.47      7290
#    macro avg       0.46      0.54      0.43      7290
# weighted avg       0.68      0.47      0.50      7290

# Recall (quantity): events caputred (?)
# Precision (quality): accuracy of prediction
# F1-score: (2 * (precision * recall)/ (precision + recall))
# Macro avg: avg of f1 scores

# --- FEATURE IMPORTANCE ---
# hr      0.262667
# eda     0.169885
# temp    0.157239
# ibi     0.153404
# bvp     0.131952
# acc     0.124853
# dtype: float32