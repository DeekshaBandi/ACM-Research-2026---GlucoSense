import pandas as pd
import numpy as np
import os

def engineer_features(p_path, p_id):
    """
    Modular feature extraction. 
    Includes multi-scale windows and temporal lags to catch glucose delays.
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
        
        cols = sdf.columns
        v_col = [c for c in cols if any(x in c for x in [s_name.lower(), 'value', 'temp', 'mag'])][0]
        
        if s_name == "ACC" and 'acc_x' in sdf.columns:
            sdf['mag'] = np.sqrt(sdf['acc_x']**2 + sdf['acc_y']**2 + sdf['acc_z']**2)
            v_col = 'mag'

        # --- CORE FEATURE MATH ---
        
        # 5-min Mean: The current state
        features_dict[f'{s_name.lower()}_mean_5'] = sdf[v_col].rolling('5min').mean()
        
        # 5-min Std: Signal volatility (Stress/Activity detection)
        features_dict[f'{s_name.lower()}_std_5'] = sdf[v_col].rolling('5min').std()
        
        # 30-min Mean: Long-term baseline context
        features_dict[f'{s_name.lower()}_mean_30'] = sdf[v_col].rolling('30min').mean()
        
        # 10-min Slope: Rate of change (Velocity)
        features_dict[f'{s_name.lower()}_slope_10'] = (sdf[v_col] - sdf[v_col].shift(1)).rolling('10min').mean()

    # --- ADVANCED TEMPORAL LAGS ---
    
    # We shift the key metrics back by 15 minutes. 
    # This correlates PAST physiology with CURRENT glucose levels.
    # Note: .shift(3) assumes a 5-minute sampling rate (3 * 5 = 15)
    lag_targets = ['hr_mean_5', 'eda_mean_5', 'acc_mean_5', 'temp_mean_5']
    for feat in lag_targets:
        if feat in features_dict:
            features_dict[f'{feat}_lag_15'] = features_dict[feat].shift(3)

    # --- CROSS-SENSOR FEATURES ---

    # HR/ACC Ratio: Helps filter out 'Physical' vs 'Metabolic' heart rate spikes
    if 'hr_mean_5' in features_dict and 'acc_mean_5' in features_dict:
        features_dict['hr_acc_ratio'] = features_dict['hr_mean_5'] / (features_dict['acc_mean_5'] + 0.1)

    return features_dict