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
    demo_path = os.path.join(data_dir, 'Demographics.csv')
    
    demo_df = pd.read_csv(demo_path)
    demo_df['ID'] = demo_df['ID'].astype(str).str.zfill(3)
    
    all_data_frames = []
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    folders.sort()
    
    print(f"Syncing 7 Features for {len(folders)} participants...")
    
    for p_id in folders:
        p_path = os.path.join(data_dir, p_id)
        label_file = os.path.join(p_path, f"Dexcom_{p_id}_labeled.csv")
        if not os.path.exists(label_file): continue
            
        try:
            # 1. Load Labels
            gdf = pd.read_csv(label_file)
            gdf['Timestamp'] = pd.to_datetime(gdf['Timestamp']).dt.floor('s')
            gdf = gdf.sort_values('Timestamp')

            # 2. Helper to load and align sensor files
            def load_sensor(name):
                f_path = os.path.join(p_path, f"{name}_{p_id}.csv")
                if not os.path.exists(f_path): return pd.DataFrame()
                df = pd.read_csv(f_path)
                df.columns = [c.strip() for c in df.columns]
                # Standardize datetime and set as index
                df['datetime'] = pd.to_datetime(df['datetime']).dt.floor('s')
                return df.set_index('datetime').sort_index()

            # 3. Load the 6 sensors
            hr = load_sensor("HR")
            temp = load_sensor("TEMP")
            eda = load_sensor("EDA")
            bvp = load_sensor("BVP")
            acc = load_sensor("ACC")
            ibi = load_sensor("IBI")

            # 4. Feature Engineering
            # Use the columns you provided for ACC
            if not acc.empty:
                acc['acc_mag'] = np.sqrt(acc['acc_x']**2 + acc['acc_y']**2 + acc['acc_z']**2)
                acc_f = acc['acc_mag'].rolling('5min').mean().to_frame('acc')
            else:
                acc_f = pd.DataFrame()

            # Rolling means for others (standard Empatica names)
            hr_f = hr['hr'].rolling('5min').mean().to_frame('hr') if not hr.empty else pd.DataFrame()
            temp_f = temp['temp'].rolling('5min').mean().to_frame('temp') if not temp.empty else pd.DataFrame()
            eda_f = eda['eda'].rolling('5min').mean().to_frame('eda') if not eda.empty else pd.DataFrame()
            bvp_f = bvp['bvp'].rolling('5min').std().to_frame('bvp') if not bvp.empty else pd.DataFrame()
            ibi_f = ibi['ibi'].rolling('5min').mean().to_frame('ibi') if not ibi.empty else pd.DataFrame()

            # 5. Master Merge
            m = gdf.copy()
            for df_feat in [hr_f, temp_f, eda_f, bvp_f, acc_f, ibi_f]:
                if not df_feat.empty:
                    m = pd.merge_asof(m, df_feat, left_on='Timestamp', right_index=True)

            # 6. Add HbA1c (7th feature)
            p_demo = demo_df[demo_df['ID'] == p_id].iloc[0]
            m['hba1c'] = p_demo['HbA1c']
            
            # Record results
            clean_m = m.dropna(subset=['hr', 'acc']) # Ensure we have the basics
            if not clean_m.empty:
                all_data_frames.append(clean_m)
                print(f"ID {p_id}: Matched {len(clean_m)} readings")
            
        except Exception as e:
            print(f"ID {p_id}: Error - {e}")

    final_df = pd.concat(all_data_frames)
    return final_df

def main():
    df = load_data()
    
    # The 7 specific features
    features = ['hr', 'temp', 'eda', 'bvp', 'acc', 'ibi', 'hba1c']
    actual_features = [f for f in features if f in df.columns]
    
    X = df[actual_features]
    le = LabelEncoder()
    y = le.fit_transform(df['Pers_Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    print(f"\nTraining 7-Feature Model on {len(X_train)} samples...")
    model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    
    print("\n" + "="*50)
    print("FINAL 7-FEATURE MODEL RESULTS")
    print("="*50)
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))
    
    importance = pd.Series(model.feature_importances_, index=actual_features).sort_values(ascending=False)
    print("\n--- SENSOR IMPORTANCE ---")
    print(importance)

if __name__ == "__main__":
    main()