import os
import pandas as pd

DAY_HOURS = range(7,23)

DAY_FEATURES = [
    'hr_10min_rolling', 'hr_30min_rolling', 'hr_60min_rolling', 'last_measured_hrv_15min',
    'hrv_change_15min', 'step_count_rollingsum_30min', 'step_count_rollingsum_60min',
    'active_energy_rollingsum_30min', 'active_energy_rollingsum_60min',
    'time_since_lastmeal_ord', 'IOB',
    'sin_hour', 'cos_hour',
    'hypoglycemia'
]

NIGHT_FEATURES = [
    'hr_10min_rolling', 'hr_30min_rolling', 'hr_60min_rolling', 'last_measured_hrv_15min',
    'hrv_change_15min', 'last_measured_ox', 'last_measured_rr', "time_since_lastmeal_ord", 'IOB', 
    'sin_hour', 'cos_hour', 'hypoglycemia'
]

def preprocess(filepath):
    df = pd.read_csv(filepath)
    is_day =df['hour'].isin(DAY_HOURS)
    day_df = df[is_day][DAY_FEATURES].reset_index(drop=True)
    night_df = df[~is_day][NIGHT_FEATURES].reset_index(drop=True)
    return day_df, night_df

def load_all(data_dir):
    participants = {}
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith('.csv'):
         continue
        pid = fname.replace('.csv', '')
        day_df, night_df = preprocess(os.path.join(data_dir, fname))
        participants[pid] = {'day': day_df, 'night': night_df}
        print(f"Participant {pid} | Day: {len(day_df)} rows | Night: {len(night_df)} rows")
    return participants


if __name__ == '__main__':
    participants = load_all('data')