import pandas as pd
import numpy as np
import os

# This script calculates and defines the Pers lavels


def process_dexcom(file_path):
    try:
        df = pd.read_csv(file_path)
        # Filter for actual glucose readings (EGV)
        glucose_df = df[df['Event Type'] == 'EGV'].copy()
        
        # Identify correct columns
        ts_col = 'Timestamp (YYYY-MM-DDThh:mm:ss)'
        val_col = 'Glucose Value (mg/dL)'
        
        if ts_col not in glucose_df.columns:
            # Fallback if names are slightly different
            ts_col = [c for c in glucose_df.columns if 'Timestamp' in c][0]
        if val_col not in glucose_df.columns:
            val_col = [c for c in glucose_df.columns if 'Glucose' in c][0]

        glucose_df = glucose_df.rename(columns={ts_col: 'Timestamp', val_col: 'Glucose'})
        
        glucose_df['Timestamp'] = pd.to_datetime(glucose_df['Timestamp'])
        glucose_df['Glucose'] = pd.to_numeric(glucose_df['Glucose'], errors='coerce')
        glucose_df = glucose_df.dropna(subset=['Glucose']).sort_values('Timestamp')
        
        # Personalized Thresholds (Rolling 24h)
        glucose_df = glucose_df.set_index('Timestamp')
        rolling = glucose_df['Glucose'].rolling(window='24h')
        
        glucose_df['Rolling_Mean'] = rolling.mean()
        glucose_df['Rolling_Std'] = rolling.std()
        
        glucose_df['Pers_Label'] = 'PersNorm'
        glucose_df.loc[glucose_df['Glucose'] > (glucose_df['Rolling_Mean'] + glucose_df['Rolling_Std']), 'Pers_Label'] = 'PersHigh'
        glucose_df.loc[glucose_df['Glucose'] < (glucose_df['Rolling_Mean'] - glucose_df['Rolling_Std']), 'Pers_Label'] = 'PersLow'
        
        return glucose_df.reset_index()
    except Exception as e:
        print(f"   Error processing data: {e}")
        return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    base_dir = os.path.join(project_root, 'data', 'glycemic_data')

    print("-" * 50)
    print("DEBUG: DATA PATH SCAN")
    print(f"Base Directory: {base_dir}")
    
    if not os.path.exists(base_dir):
        print("CRITICAL: Base directory does not exist.")
        return

    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    print(f"Found {len(folders)} folders: {folders}")
    print("-" * 50)

    summary_stats = []

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        # Look for any file that starts with 'Dexcom' and ends with '.csv'
        files_in_folder = os.listdir(folder_path)
        dexcom_file = [f for f in files_in_folder if f.startswith('Dexcom') and f.endswith('.csv')]
        
        if not dexcom_file:
            print(f"Skipping {folder}: No Dexcom CSV found. Files seen: {files_in_folder}")
            continue
            
        # Use the first Dexcom file found
        file_to_open = os.path.join(folder_path, dexcom_file[0])
        print(f"Processing: {folder}/{dexcom_file[0]}...")
        
        labeled_df = process_dexcom(file_to_open)
        
        if labeled_df is not None:
            output_name = dexcom_file[0].replace(".csv", "_labeled.csv")
            output_path = os.path.join(folder_path, output_name)
            labeled_df.to_csv(output_path, index=False)
            
            counts = labeled_df['Pers_Label'].value_counts().to_dict()
            summary_stats.append({
                'ID': folder,
                'High': counts.get('PersHigh', 0),
                'Low': counts.get('PersLow', 0)
            })
    
    if summary_stats:
        print("\nSUMMARY TABLE")
        print(pd.DataFrame(summary_stats).to_string(index=False))

if __name__ == "__main__":
    main()