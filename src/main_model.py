import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score # Fixed import here
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import warnings

warnings.filterwarnings("ignore")

def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, 'data', 'glycemic_data')
    
    all_data_frames = []
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    print(f"Syncing Balanced Passive Streams + Trends for {len(folders)} participants...")
    
    for p_id in folders:
        p_path = os.path.join(data_dir, p_id)
        label_file = os.path.join(p_path, f"Dexcom_{p_id}_labeled.csv")
        if not os.path.exists(label_file): continue
            
        try:
            # 1. Load Labels
            gdf = pd.read_csv(label_file)
            gdf['Timestamp'] = pd.to_datetime(gdf['Timestamp']).dt.floor('s')
            # Handle duplicates (ID 001/015 fix)
            gdf = gdf.sort_values('Timestamp').groupby('Timestamp').first() 

            # 2. Optimized Sensor Loading
            def load_sensor_fast(name):
                f_path = os.path.join(p_path, f"{name}_{p_id}.csv")
                if not os.path.exists(f_path): return pd.DataFrame()
                df = pd.read_csv(f_path)
                df.columns = [c.strip().lower() for c in df.columns]
                t_col = 'datetime' if 'datetime' in df.columns else 'time'
                df[t_col] = pd.to_datetime(df[t_col]).dt.floor('s')
                return df.groupby(t_col).mean().sort_index()

            sensors = {s: load_sensor_fast(s) for s in ["HR", "TEMP", "EDA", "BVP", "ACC", "IBI"]}

            # 3. Feature Engineering: Means AND Trends (Slopes)
            features_dict = {}
            for s_name, sdf in sensors.items():
                if sdf.empty: continue
                
                # Dynamic column finding
                cols = sdf.columns
                v_col = [c for c in cols if any(x in c for x in [s_name.lower(), 'value', 'temp', 'mag'])][0]
                
                # Special handling for ACC Magnitude
                if s_name == "ACC" and 'acc_x' in sdf.columns:
                    sdf['mag'] = np.sqrt(sdf['acc_x']**2 + sdf['acc_y']**2 + sdf['acc_z']**2)
                    v_col = 'mag'

                # FEATURE: 5-min Mean (State)
                features_dict[f'{s_name.lower()}_mean'] = sdf[v_col].rolling('5min').mean()
                # FEATURE: 10-min Slope (Trend/Velocity)
                features_dict[f'{s_name.lower()}_slope'] = (sdf[v_col] - sdf[v_col].shift(1)).rolling('10min').mean()

            # 4. Temporal Alignment
            m = gdf.copy()
            for name, series in features_dict.items():
                m = pd.merge_asof(m, series.to_frame(name), left_index=True, right_index=True)

            # Drop rows missing core sensor data
            clean_m = m.dropna(subset=['hr_mean', 'eda_mean', 'temp_mean'])
            if not clean_m.empty:
                all_data_frames.append(clean_m)
                print(f"ID {p_id}: Synced {len(clean_m)} windows")
            
        except Exception as e:
            print(f"ID {p_id}: Error - {e}")

    return pd.concat(all_data_frames)

def main():
    df = load_data()
    
    # 5. CLASS BALANCING 
    # Undersampling the majority 'PersNorm' class
    counts = df['Pers_Label'].value_counts()
    min_class_size = counts.min() # Usually PersHigh or PersLow
    
    df_high = df[df['Pers_Label'] == 'PersHigh']
    df_low = df[df['Pers_Label'] == 'PersLow']
    df_norm = df[df['Pers_Label'] == 'PersNorm']
    
    # Balance all to the size of the smallest class (e.g., PersLow)
    df_high_bal = resample(df_high, replace=False, n_samples=min_class_size, random_state=42)
    df_low_bal = resample(df_low, replace=False, n_samples=min_class_size, random_state=42)
    df_norm_bal = resample(df_norm, replace=False, n_samples=min_class_size, random_state=42)
    
    df_balanced = pd.concat([df_high_bal, df_low_bal, df_norm_bal])
    
    print(f"\nDataset Balanced: {min_class_size} samples per class.")

    # 6. Training
    features = [c for c in df_balanced.columns if '_mean' in c or '_slope' in c]
    X = df_balanced[features]
    le = LabelEncoder()
    y = le.fit_transform(df_balanced['Pers_Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # XGBoost Config
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    # 7. Results
    print("\n" + "="*50)
    print("BALANCED PASSIVE SENSOR RESULTS")
    print("="*50)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Balanced Accuracy Check
    acc = accuracy_score(y_test, y_pred)
    print(f"Overall Balanced Accuracy: {acc:.4f}")
    
    # Feature Importance for minimization
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n--- FEATURE IMPORTANCE (Minimal Set Candidates) ---")
    print(importance)

if __name__ == "__main__":
    main()