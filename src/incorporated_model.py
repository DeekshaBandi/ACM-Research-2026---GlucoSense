import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

def load_hybrid_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, 'data', 'glycemic_data')
    
    all_data = []
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    print(f"Syncing Full Sensor Fusion for {len(folders)} participants...")
    
    for p_id in folders:
        p_path = os.path.join(data_dir, p_id)
        files = {
            'label': f"Dexcom_{p_id}_labeled.csv",
            'hr': f"HR_{p_id}.csv",
            'eda': f"EDA_{p_id}.csv",
            'temp': f"TEMP_{p_id}.csv",
            'food': f"Food_Log_{p_id}.csv"
        }
        
        # Verify core files exist
        if not all(os.path.exists(os.path.join(p_path, f)) for f in [files['label'], files['hr'], files['food']]):
            continue
            
        try:
            # 1. LOAD LABELS & FIX DUPLICATES (Fixes ID 001 & 015)
            gdf = pd.read_csv(os.path.join(p_path, files['label']))
            gdf.columns = [c.strip() for c in gdf.columns]
            gdf['Timestamp'] = pd.to_datetime(gdf['Timestamp'])
            # CRITICAL FIX: Group by timestamp and take the mean/first to remove duplicates
            gdf = gdf.groupby('Timestamp').first().sort_index()

            # 2. LOAD SENSORS (HR, EDA, TEMP)
            feats = {}
            for s_key in ['hr', 'eda', 'temp']:
                s_file = os.path.join(p_path, files[s_key])
                if not os.path.exists(s_file): continue
                
                sdf = pd.read_csv(s_file)
                sdf.columns = [c.strip().lower() for c in sdf.columns]
                
                # Dynamic column finding
                t_col = [c for c in sdf.columns if 'time' in c][0]
                v_col = [c for c in sdf.columns if any(x in c for x in [s_key, 'value', 'temp'])][0]
                
                sdf[t_col] = pd.to_datetime(sdf[t_col])
                sdf = sdf.groupby(t_col)[v_col].mean().sort_index() # Remove duplicates here too
                
                # Feature: 10-min rolling mean (smooths sensor jitter)
                feats[f'{s_key}_mean'] = sdf.rolling('10min').mean().reindex(gdf.index, method='ffill')
                # Feature: 20-min Trend (is the sensor value rising or falling?)
                feats[f'{s_key}_trend'] = (sdf - sdf.shift(1)).rolling('20min').mean().reindex(gdf.index, method='ffill')

            # 3. LOAD FOOD (Carbs on Board)
            fdf = pd.read_csv(os.path.join(p_path, files['food']))
            fdf.columns = [c.strip().lower() for c in fdf.columns]
            # Standardizing food time column
            if 'time_begin' in fdf.columns:
                fdf['dt'] = pd.to_datetime(fdf['time_begin'], errors='coerce')
            elif 'date' in fdf.columns:
                t_col = 'time' if 'time' in fdf.columns else 'time_of_day'
                fdf['dt'] = pd.to_datetime(fdf['date'].astype(str) + ' ' + fdf[t_col].astype(str), errors='coerce')
            else:
                fdf = pd.read_csv(os.path.join(p_path, files['food']), header=None)
                fdf['dt'] = pd.to_datetime(fdf[0].astype(str) + ' ' + fdf[1].astype(str), errors='coerce')
                fdf = fdf.rename(columns={9: 'total_carb'})

            fdf = fdf.dropna(subset=['dt']).groupby('dt')['total_carb'].sum().sort_index()
            # Feature: Carbs consumed in last 3 hours (Glycemic impact duration)
            feats['COB_3hr'] = fdf.rolling('180min').sum().reindex(gdf.index, method='ffill').fillna(0)

            # 4. MERGE
            combined = pd.DataFrame(feats)
            combined['Label'] = gdf['Pers_Label']
            combined = combined.dropna()
            
            all_data.append(combined)
            print(f"ID {p_id}: Successfully synced {len(combined)} windows")

        except Exception as e:
            print(f"ID {p_id}: Error - {e}")

    return pd.concat(all_data)

def main():
    df = load_hybrid_data()
    
    # We now have 7 features: mean + trend for HR/EDA/TEMP, plus Food
    features = [c for c in df.columns if c != 'Label']
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    # XGBoost with slight regularization to handle the noise of 16 different bodies
    model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.05, 
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, sample_weight=weights)
    
    print("\n" + "="*50)
    print("STRATEGY 4: FULL SENSOR FUSION + TRENDS + FOOD")
    print("="*50)
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

if __name__ == "__main__":
    main()


# ------ OUTPUT ------
# ID 001: Successfully synced 2561 windows
# ID 002: Successfully synced 2119 windows
# ID 003: Successfully synced 2214 windows
# ID 004: Successfully synced 2164 windows
# ID 005: Successfully synced 2557 windows
# ID 006: Successfully synced 2847 windows
# ID 007: Successfully synced 2207 windows
# ID 008: Successfully synced 2505 windows
# ID 009: Successfully synced 2306 windows
# ID 010: Successfully synced 2148 windows
# ID 011: Successfully synced 2843 windows
# ID 012: Successfully synced 2169 windows
# ID 013: Successfully synced 1979 windows
# ID 014: Successfully synced 1878 windows
# ID 015: Successfully synced 1673 windows
# ID 016: Successfully synced 2277 windows

# ==================================================
# STRATEGY 3: FULL SENSOR FUSION + TRENDS + FOOD
# ==================================================
#               precision    recall  f1-score   support

#     PersHigh       0.35      0.67      0.46      1117
#      PersLow       0.32      0.69      0.44      1034
#     PersNorm       0.90      0.51      0.65      5139

#     accuracy                           0.56      7290
#    macro avg       0.52      0.62      0.52      7290
# weighted avg       0.73      0.56      0.59      7290