import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

def load_food_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(project_root, 'data', 'glycemic_data')
    
    all_data_frames = []
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    folders.sort()
    
    print(f"Syncing Food Logs for {len(folders)} participants (including ID 003 fix)...")
    
    for p_id in folders:
        p_path = os.path.join(data_dir, p_id)
        label_file = os.path.join(p_path, f"Dexcom_{p_id}_labeled.csv")
        food_file = os.path.join(p_path, f"Food_Log_{p_id}.csv")
        
        if not os.path.exists(label_file) or not os.path.exists(food_file):
            continue
            
        try:
            gdf = pd.read_csv(label_file)
            gdf['Timestamp'] = pd.to_datetime(gdf['Timestamp'])
            
            # --- START FIX FOR ID 003 ---
            # Peek at the file to see if it has a header
            peek = pd.read_csv(food_file, nrows=1)
            
            if 'date' in peek.columns or 'time_begin' in peek.columns:
                # Standard file with headers
                fdf = pd.read_csv(food_file)
                fdf.columns = [c.strip().lower() for c in fdf.columns]
                # Fallback: if time_begin is missing but date/time are there
                if 'time_begin' not in fdf.columns:
                    time_col = 'time' if 'time' in fdf.columns else 'time_of_day'
                    fdf['dt'] = pd.to_datetime(fdf['date'].astype(str) + ' ' + fdf[time_col].astype(str), errors='coerce')
                else:
                    fdf['dt'] = pd.to_datetime(fdf['time_begin'], errors='coerce')
            else:
                # ID 003 Logic: No headers found
                fdf = pd.read_csv(food_file, header=None)
                # Position 0 = Date, Position 1 = Time
                fdf['dt'] = pd.to_datetime(fdf[0].astype(str) + ' ' + fdf[1].astype(str), errors='coerce')
                # Map nutrient columns by index (Calorie=8, Carb=9, Sugar=11)
                fdf = fdf.rename(columns={8: 'calorie', 9: 'total_carb', 11: 'sugar'})
            # --- END FIX ---

            fdf = fdf.dropna(subset=['dt'])
            nutrient_cols = [c for c in ['total_carb', 'sugar', 'calorie'] if c in fdf.columns]
            fdf = fdf.groupby('dt')[nutrient_cols].sum().sort_index()

            food_features = pd.DataFrame(index=gdf['Timestamp'])
            for col in nutrient_cols:
                rolled = fdf[col].rolling('120min').sum()
                food_features[f'{col}_2hr'] = rolled.reindex(gdf['Timestamp'], method='ffill').fillna(0)

            combined = pd.concat([gdf.set_index('Timestamp'), food_features], axis=1).reset_index()
            combined = combined.dropna(subset=['Pers_Label'])
            all_data_frames.append(combined)
            print(f"ID {p_id}: Successfully processed")
            
        except Exception as e:
            print(f"ID {p_id}: Error - {e}")

    return pd.concat(all_data_frames)

def main():
    df = load_food_data()
    features = [c for c in df.columns if '_2hr' in c]
    
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df['Pers_Label'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights)
    
    print("\n" + "="*50)
    print("STRATEGY 2: FOOD-ONLY RESULTS (ALL 16 PATIENTS)")
    print("="*50)
    print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

if __name__ == "__main__":
    main()

# ------ OUTPUT ------
# Syncing Food Logs for 16 participants (including ID 003 fix)...
# ID 001: Successfully processed
# ID 002: Successfully processed
# ID 003: Successfully processed
# ID 004: Successfully processed
# ID 005: Successfully processed
# ID 006: Successfully processed
# ID 007: Successfully processed
# ID 008: Successfully processed
# ID 009: Successfully processed
# ID 010: Successfully processed
# ID 011: Successfully processed
# ID 012: Successfully processed
# ID 013: Successfully processed
# ID 014: Successfully processed
# ID 015: Successfully processed
# ID 016: Successfully processed

# ==================================================
# STRATEGY 2: FOOD-ONLY RESULTS (ALL 16 PATIENTS)
# ==================================================
#               precision    recall  f1-score   support

#     PersHigh       0.29      0.43      0.35      1140
#      PersLow       0.22      0.75      0.34      1051
#     PersNorm       0.84      0.36      0.50      5189

#     accuracy                           0.42      7380
#    macro avg       0.45      0.51      0.40      7380
# weighted avg       0.67      0.42      0.45      7380