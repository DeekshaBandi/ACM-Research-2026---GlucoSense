import pandas as pd
import numpy as np
import os

def engineer_features(p_path, p_id):
    """
    Main feature extraction pipeline. 
    Returns a dict of series keyed by feature name for easy merging.
    """
    def load_sensor(name):
        f_path = os.path.join(p_path, f"{name}_{p_id}.csv")
        if not os.path.exists(f_path): return pd.DataFrame()
        df = pd.read_csv(f_path)
        df.columns = [c.strip().lower() for c in df.columns]
        t_col = 'datetime' if 'datetime' in df.columns else 'time'
        df[t_col] = pd.to_datetime(df[t_col]).dt.floor('s')
        return df.groupby(t_col).mean().sort_index()

    sensors = {s: load_sensor(s) for s in ["HR", "TEMP", "EDA", "BVP", "ACC", "IBI"]}
    
    features_dict = {}
    for s_name, sdf in sensors.items():
        if sdf.empty: continue
        
        # Helper to grab the actual data column since names vary (e.g., 'temp' vs 'value')
        cols = sdf.columns
        v_col = [c for c in cols if any(x in c for x in [s_name.lower(), 'value', 'temp', 'mag'])][0]
        
        # Convert raw XYZ to G-force magnitude to make movement orientation-agnostic
        if s_name == "ACC" and 'acc_x' in sdf.columns:
            sdf['mag'] = np.sqrt(sdf['acc_x']**2 + sdf['acc_y']**2 + sdf['acc_z']**2)
            v_col = 'mag'

        # --- FEATURE EXTRACTION BLOCK ---

        # 5-min Mean: Gets the current physiological 'state' 
        features_dict[f'{s_name.lower()}_mean_5'] = sdf[v_col].rolling('5min', closed='left', min_periods=1).mean()
        
        # 5-min Std: Captures signal 'jitter' or instability (big for EDA/Stress)
        features_dict[f'{s_name.lower()}_std_5'] = sdf[v_col].rolling('5min', closed='left', min_periods=1).std()
        
        # 30-min Mean: Provides long-term context to filter out momentary noise
        features_dict[f'{s_name.lower()}_mean_30'] = sdf[v_col].rolling('30min', closed='left', min_periods=1).mean()
        
        # 10-min Slope: Calculates velocity of change (is the signal spiking or crashing?)
        features_dict[f'{s_name.lower()}_slope_10'] = (sdf[v_col] - sdf[v_col].shift(1)).rolling('10min', closed='left', min_periods=1).mean()

    # --- CROSS-SENSOR LOGIC ---

    # HR-to-ACC Ratio: Isolates 'Resting' HR spikes from 'Exercise' HR spikes
    # Adding 0.1 to denominator to avoid division by zero errors
    if 'hr_mean_5' in features_dict and 'acc_mean_5' in features_dict:
        features_dict['hr_acc_ratio'] = features_dict['hr_mean_5'] / (features_dict['acc_mean_5'] + 0.1)

    return features_dict