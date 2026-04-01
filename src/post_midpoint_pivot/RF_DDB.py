import pandas as pd
import numpy as np
import os
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.impute import SimpleImputer

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data_new/"))
OUTPUT_DIR = os.path.abspath(os.path.join(DATA_DIR, "RF_results/"))
LABEL_FILE = os.path.join(DATA_DIR, "cgm_daily_labels.csv")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def load_and_merge_data():
    # 1. Load Labels
    labels_df = pd.read_csv(LABEL_FILE)
    labels_df['participant'] = labels_df['participant'].astype(str).str.zfill(3)
    labels_df['date'] = pd.to_datetime(labels_df['date']).dt.date

    # 2. Load Feature Files
    all_features = []
    for file in os.listdir(DATA_DIR):
        if file.startswith("P") and file.endswith(".csv"):
            df = pd.read_csv(os.path.join(DATA_DIR, file))
            df['Subject_ID'] = df['Subject_ID'].astype(str).str.zfill(3)
            df['Date'] = pd.to_datetime(df['Date']).dt.date
            all_features.append(df)
    
    features_df = pd.concat(all_features, ignore_index=True)

    # 3. Merge
    merged = pd.merge(
        features_df, 
        labels_df[['participant', 'date', 'label']], 
        left_on=['Subject_ID', 'Date'], 
        right_on=['participant', 'date']
    )
    return merged

def get_sensor_groups(columns):
    return {
        "Cardiac": [c for c in columns if c.startswith(('HR_', 'IBI_', 'BVP_'))],
        "ACC": [c for c in columns if c.startswith('ACC_')],
        "EDA": [c for c in columns if c.startswith('EDA_')],
        "TEMP": [c for c in columns if c.startswith('TEMP_')]
    }

def run_ablation_study():
    data = load_and_merge_data()
    groups_dict = get_sensor_groups(data.columns)
    modalities = list(groups_dict.keys())
    
    X_full = data.drop(columns=['label', 'participant', 'date', 'Date', 'Subject_ID', 'Window_Start', 'Window_End'], errors='ignore')
    y = data['label']
    subjects = data['Subject_ID'] # Used for LOSO grouping
    
    imputer = SimpleImputer(strategy='mean')
    logo = LeaveOneGroupOut() # The LOSO engine
    
    master_results = []

    print(f"Starting LOSO ablation study with {subjects.nunique()} participants...")

    for r in range(1, 5):
        for combo in itertools.combinations(modalities, r):
            combo_features = []
            for m in combo:
                combo_features.extend(groups_dict[m])
            
            X = data[combo_features]
            X_imputed = imputer.fit_transform(X)

            # Initialize Model
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Run LOSO Cross-Validation
            # groups=subjects ensures it leaves one whole person out at a time
            f1_scores = cross_val_score(rf, X_imputed, y, groups=subjects, cv=logo, scoring='f1_macro')
            avg_f1 = np.mean(f1_scores)

            # --- Naming and Saving ---
            # New format: name1-name2-name3_RF.csv
            combo_name = "-".join(combo)
            filename = f"{combo_name}_RF.csv"
            
            summary_data = {
                "Combination": combo_name,
                "Sensors": ", ".join(combo),
                "LOSO_F1_Macro": avg_f1,
                "Num_Participants": subjects.nunique(),
                "Total_Days": len(data)
            }
            
            pd.DataFrame([summary_data]).to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
            master_results.append(summary_data)
            print(f"[SUCCESS] {filename}: LOSO F1 = {avg_f1:.4f}")

    # Generate the Master Summary for easy comparison
    master_df = pd.DataFrame(master_results).sort_values(by="LOSO_F1_Macro", ascending=False)
    master_df.to_csv(os.path.join(OUTPUT_DIR, "Ablation_Master_Summary.csv"), index=False)

if __name__ == "__main__":

    run_ablation_study()
    print("\n" + "="*30)
    print(f"LOSO STUDY COMPLETE")
    print(f"Check results in: {OUTPUT_DIR}")
    print("="*30)


# --------- Initial output ---------

# [SUCCESS] Cardiac_RF.csv: LOSO F1 = 0.4604
# [SUCCESS] ACC_RF.csv: LOSO F1 = 0.4637
# [SUCCESS] EDA_RF.csv: LOSO F1 = 0.3397
# [SUCCESS] TEMP_RF.csv: LOSO F1 = 0.4923
# [SUCCESS] Cardiac-ACC_RF.csv: LOSO F1 = 0.4311
# [SUCCESS] Cardiac-EDA_RF.csv: LOSO F1 = 0.4077
# [SUCCESS] Cardiac-TEMP_RF.csv: LOSO F1 = 0.5394
# [SUCCESS] ACC-EDA_RF.csv: LOSO F1 = 0.3539
# [SUCCESS] ACC-TEMP_RF.csv: LOSO F1 = 0.4286
# [SUCCESS] EDA-TEMP_RF.csv: LOSO F1 = 0.3761
# [SUCCESS] Cardiac-ACC-EDA_RF.csv: LOSO F1 = 0.4219
# [SUCCESS] Cardiac-ACC-TEMP_RF.csv: LOSO F1 = 0.4847
# [SUCCESS] Cardiac-EDA-TEMP_RF.csv: LOSO F1 = 0.4644
# [SUCCESS] ACC-EDA-TEMP_RF.csv: LOSO F1 = 0.4416
# [SUCCESS] Cardiac-ACC-EDA-TEMP_RF.csv: LOSO F1 = 0.4420

# ------------------------------------------------------