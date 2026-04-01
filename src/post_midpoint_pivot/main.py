import pandas as pd
import numpy as np
import os
from datetime import timedelta


# ---- BRIEF ----
# We iterate through each patient file and we aim to to create a new csv for each patient.
# Each of the new tables should have row(s) of participant-day information.
# The columns are four metrics for each sensor: mean, std, 90th and 10th percentile



# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data/glycemic_data/"))
OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../data_new/"))
MIN_HOURS = 5

def calculate_rmssd(ibi_series):
    """Calculates Root Mean Square of Successive Differences for HRV."""
    if len(ibi_series) < 2:
        return np.nan
    diffs = np.diff(ibi_series)
    return np.sqrt(np.mean(diffs**2))

def extract_sensor_stats(df, sensor_name):
    """Calculates the 4 core statistical signatures + IBI specials."""
    # Define the 4 standard metrics we want for every sensor
    metric_names = ['mean', 'std', '90th', '10th']
    
    if df is None or df.empty:
        return {f"{sensor_name}_{m}": np.nan for m in metric_names}
    
    # Use the last column as the data value
    val_col = df.columns[-1]
    data = df[val_col]
    
    stats = {
        f"{sensor_name}_mean": data.mean(),
        f"{sensor_name}_std": data.std(),
        f"{sensor_name}_90th": data.quantile(0.9),
        f"{sensor_name}_10th": data.quantile(0.1)
    }
    
    # Add RMSSD specifically for IBI
    if sensor_name.upper() == "IBI":
        stats["IBI_rmssd"] = calculate_rmssd(data)
        
    return stats

def process_participant(pid):
    participant_path = os.path.join(DATA_DIR, pid)
    
    # CORE: Day is valid only if these overlap for 5+ hours
    required_sensors = ['HR', 'IBI', 'EDA', 'ACC', 'TEMP']
    # OPTIONAL: Extract if present, but don't delete day if missing
    optional_sensors = ['BVP']
    
    all_sensors = required_sensors + optional_sensors
    sensor_data = {}
    
    # 1. Load Data
    for sensor in all_sensors:
        file_path = os.path.join(participant_path, f"{sensor}_{pid}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['datetime'] = pd.to_datetime(df.iloc[:, 0], unit='s')
            sensor_data[sensor] = df
        elif sensor in required_sensors:
            print(f"Skipping {pid}: Missing REQUIRED sensor {sensor}")
            return 0

    # 2. Identify common dates across REQUIRED sensors
    common_dates = set(sensor_data[required_sensors[0]]['datetime'].dt.date)
    for s in required_sensors[1:]:
        common_dates &= set(sensor_data[s]['datetime'].dt.date)
    
    daily_features = []

    # 3. Process dates
    for date in sorted(common_dates):
        starts, ends = [], []
        day_clips = {}
        
        # Clip all available sensors for this date
        for sensor in all_sensors:
            if sensor in sensor_data:
                df = sensor_data[sensor]
                clip = df[df['datetime'].dt.date == date]
                if not clip.empty:
                    day_clips[sensor] = clip
                    if sensor in required_sensors:
                        starts.append(clip['datetime'].min())
                        ends.append(clip['datetime'].max())
        
        # Sync window based on the "tightest" overlap of required sensors
        sync_start = max(starts)
        sync_end = min(ends)
        duration = sync_end - sync_start
        
        # 4. Gatekeeper
        if duration >= timedelta(hours=MIN_HOURS):
            day_row = {
                "Subject_ID": pid,
                "Date": date,
                "Window_Start": sync_start,
                "Window_End": sync_end,
                "Overlap_Hours": duration.total_seconds() / 3600
            }
            
            # 5. Feature Extraction for all sensors
            for sensor in all_sensors:
                # If the sensor exists for this participant AND has data in this day
                if sensor in day_clips:
                    clip = day_clips[sensor]
                    # Further clip data to the exact synchronized window
                    sync_clip = clip[(clip['datetime'] >= sync_start) & (clip['datetime'] <= sync_end)]
                    day_row.update(extract_sensor_stats(sync_clip, sensor))
                else:
                    # Fill NaNs if BVP is missing for this participant or specific day
                    day_row.update({f"{sensor}_{m}": np.nan for m in ['mean', 'std', '90th', '10th']})
            
            daily_features.append(day_row)
        else:
            print(f"Discarding {pid} on {date}: Insufficient overlap ({duration})")

    # 6. Save results
    if daily_features:
        output_df = pd.DataFrame(daily_features)
        output_path = os.path.join(OUTPUT_DIR, f"P{pid}.csv")
        output_df.to_csv(output_path, index=False)
        print(f"[SUCCESS] Saved P{pid}.csv with {len(daily_features)} days.")
        return len(daily_features)
    
    return 0

if __name__ == "__main__":
    # Create directory if it doesn't exist (but don't wipe)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    total_days = 0
    participants = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    print(f"Starting BVP-Optional extraction for {len(participants)} participants...")
    for p in sorted(participants):
        total_days += process_participant(p)
    
    print("\n" + "="*30)
    print(f"FINAL AUDIT COMPLETE")
    print(f"Total Participant-Days Engineered: {total_days}")
    print("="*30)


# Initial Output

# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P001.csv with 10 days.
# --- Finished Participant 001 ---
# Total Valid Days: 10
# ------------------------------
# Discarding 002 on 2020-02-29: Only 0 days 04:37:54.263252 of overlap.
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P002.csv with 8 days.
# --- Finished Participant 002 ---
# Total Valid Days: 8
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P003.csv with 8 days.
# --- Finished Participant 003 ---
# Total Valid Days: 8
# ------------------------------
# Discarding 004 on 2020-02-28: Only 0 days 04:14:19.354736 of overlap.
# Discarding 004 on 2020-02-29: Only 0 days 04:51:39.574764 of overlap.
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P004.csv with 7 days.
# --- Finished Participant 004 ---
# Total Valid Days: 7
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P005.csv with 10 days.
# --- Finished Participant 005 ---
# Total Valid Days: 10
# ------------------------------
# Discarding 006 on 2020-03-09: Only 0 days 02:04:54.405553 of overlap.
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P006.csv with 8 days.
# --- Finished Participant 006 ---
# Total Valid Days: 8
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P007.csv with 9 days.
# --- Finished Participant 007 ---
# Total Valid Days: 9
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P008.csv with 10 days.
# --- Finished Participant 008 ---
# Total Valid Days: 10
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P009.csv with 9 days.
# --- Finished Participant 009 ---
# Total Valid Days: 9
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P010.csv with 9 days.
# --- Finished Participant 010 ---
# Total Valid Days: 9
# ------------------------------
# Discarding 011 on 2020-04-16: Only 0 days 03:51:58.934007 of overlap.
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P011.csv with 9 days.
# --- Finished Participant 011 ---
# Total Valid Days: 9
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P012.csv with 9 days.
# --- Finished Participant 012 ---
# Total Valid Days: 9
# ------------------------------
# Discarding 013 on 2020-06-06: Only 0 days 00:39:32.483598 of overlap.
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P013.csv with 9 days.
# --- Finished Participant 013 ---
# Total Valid Days: 9
# ------------------------------
# Discarding 014 on 2020-06-06: Only 0 days 01:24:14.354630 of overlap.
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P014.csv with 7 days.
# --- Finished Participant 014 ---
# Total Valid Days: 7
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P015.csv with 8 days.
# --- Finished Participant 015 ---
# Total Valid Days: 8
# ------------------------------
# Successfully saved c:\Users\ericl\Downloads\GlucoSense\ACM-Research-2026---GlucoSense\data_new\P016.csv with 8 days.
# --- Finished Participant 016 ---
# Total Valid Days: 8
# ------------------------------


# Latest output

# Starting BVP-Optional extraction for 16 participants...
# [SUCCESS] Saved P001.csv with 10 days.
# Discarding 002 on 2020-02-29: Insufficient overlap (0 days 04:37:54.263252)
# [SUCCESS] Saved P002.csv with 8 days.
# [SUCCESS] Saved P003.csv with 8 days.
# Discarding 004 on 2020-02-28: Insufficient overlap (0 days 04:14:19.354736)
# Discarding 004 on 2020-02-29: Insufficient overlap (0 days 04:51:39.574764)
# [SUCCESS] Saved P004.csv with 7 days.
# [SUCCESS] Saved P005.csv with 10 days.
# Discarding 006 on 2020-03-09: Insufficient overlap (0 days 02:04:54.405553)
# [SUCCESS] Saved P006.csv with 8 days.
# [SUCCESS] Saved P007.csv with 9 days.
# [SUCCESS] Saved P008.csv with 10 days.
# [SUCCESS] Saved P009.csv with 9 days.
# [SUCCESS] Saved P010.csv with 9 days.
# Discarding 011 on 2020-04-16: Insufficient overlap (0 days 03:51:58.934007)
# [SUCCESS] Saved P011.csv with 9 days.
# [SUCCESS] Saved P012.csv with 9 days.
# Discarding 013 on 2020-06-06: Insufficient overlap (0 days 00:39:32.483598)
# [SUCCESS] Saved P013.csv with 9 days.
# Discarding 014 on 2020-06-06: Insufficient overlap (0 days 01:24:14.354630)
# [SUCCESS] Saved P014.csv with 7 days.
# [SUCCESS] Saved P015.csv with 8 days.
# [SUCCESS] Saved P016.csv with 8 days.

# ==============================
# FINAL AUDIT COMPLETE
# Total Participant-Days Engineered: 138
# ==============================
