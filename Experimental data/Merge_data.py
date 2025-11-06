import pandas as pd
from pathlib import Path
import re
import numpy as np

# --- 1. Configuration ---

# The script assumes it is in the same folder as 'Original_experimental_data'
# BASE_PATH points to the 'Original_experimental_data' folder
BASE_PATH = Path("Original_experimental_data")

# OUTPUT_PATH is the directory ('Merged_experimental_data')
OUTPUT_PATH = Path("Merged_experimental_data")

# Define the scenarios to process
SCENARIOS = {
    "OCC scenario": "OCC_Scenario_Merged_Data.csv",
    "Baseline scenario": "Baseline_Scenario_Merged_Data.csv"
}

# Define the interpolation time limit (1 hour)
INTERPOLATE_LIMIT_MINS = 60 

# --- 2. Helper Function: Get Sensor File Targets ---

def get_sensor_targets():
    """
    Defines all sensor files to be loaded, their time columns, and how to rename them.
    """
    targets = []
    
    # AC (1-3)
    for i in range(1, 4):
        targets.append((
            f"ac_sensor_data/ac{i}_settings_data.csv", 'Time',
            { 'Switch_State': f'Switch_State_ac{i}', 'Wind_Speed': f'Wind_Speed_ac{i}', 'Setpoint': f'Setpoint_ac{i}' }
        ))
        targets.append((
            f"ac_sensor_data/ac{i}_energy_consumption_data.csv", 'Time',
            { 'Energy_Consumption(Wh)': f'Energy_Consumption(Wh)_ac{i}' }
        ))
        
    # Light Illuminance (1-8)
    for i in range(1, 9):
        targets.append((
            f"light_illuminance_data/light_illuminance{i}.csv", 'Time',
            { 'Illuminance(lux)': f'Illuminance(lux)_light{i}' }
        ))
        
    # Light State (1-8)
    for i in range(1, 9):
        targets.append((
            f"light_sensor_data/light_switch{i}.csv", 'Time',
            { 'State': f'Light_state_{i}' }  # Renamed as requested
        ))
        
    # Socket State (1-8)
    for i in range(1, 9):
        targets.append((
            f"socket_sensor_data/socket_switch{i}.csv", 'Time',
            { 'State': f'Socket_state_{i}' } # Renamed as requested
        ))
        
    # Switch State (1-8)
    for i in range(1, 9):
        targets.append((
            f"switch_sensor_data/switch_sensor{i}.csv", 'Time',
            { 'Switch_state': f'Wireless_switch_{i}' } # Renamed as requested
        ))
        
    # Temp (1-8)
    for i in range(1, 9):
        targets.append((
            f"temp_sensor_data/temp_sensor{i}.csv", 'Time',
            { 'Indoor_temperature': f'Workstation_temperature_{i}' }
        ))
        
    return targets

# --- 3. Helper Function: Load and Prep Sensor Data ---

def load_and_prep_sensor(file_path, time_col, rename_map):
    """
    Loads a single sensor CSV, prepares it for a minute-level merge.
    """
    if not file_path.exists():
        print(f"    - WARNING: File not found {file_path.name}, skipping.")
        return None

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            print(f"    - WARNING: File is empty {file_path.name}, skipping.")
            return None

        # Check for required columns
        required_cols = list(rename_map.keys()) + [time_col]
        if not all(col in df.columns for col in required_cols):
            print(f"    - WARNING: File {file_path.name} is missing required columns, skipping.")
            return None

        # --- Time Alignment ---
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df = df.dropna(subset=[time_col])
        if df.empty:
            return None
        
        df['join_key'] = df[time_col].dt.floor('T')
        df = df.sort_values(by=time_col).drop_duplicates(subset=['join_key'], keep='last')
        
        columns_to_keep = list(rename_map.keys()) + ['join_key']
        df_to_merge = df[columns_to_keep].rename(columns=rename_map)
        
        return df_to_merge

    except Exception as e:
        print(f"    - ERROR: Failed to process {file_path.name}: {e}")
        return None

# --- 4. Main Processing Function for a Scenario ---

def process_scenario(scenario_name, scenario_path, output_file_path):
    """
    Runs the full merge and interpolation pipeline for one scenario.
    """
    print(f"\n--- Processing Scenario: {scenario_name} ---")

    # 1. Load Master File (occupied_number.csv)
    master_file = scenario_path / "occupied_number.csv"
    if not master_file.exists():
        print(f"ERROR: Master file not found {master_file.name}. Skipping this scenario.")
        return

    try:
        occupied_df = pd.read_csv(master_file)
        occupied_df['datetime'] = pd.to_datetime(occupied_df['datetime'])
        occupied_df = occupied_df.dropna(subset=['datetime'])
        occupied_df['join_key'] = occupied_df['datetime'].dt.floor('T')
        occupied_df = occupied_df.drop_duplicates(subset=['join_key'], keep='last')
        
    except Exception as e:
        print(f"ERROR: Could not read master file {master_file.name}: {e}")
        return

    # 2. Create Master 1-Minute Time Index
    min_time = occupied_df['datetime'].min().floor('T')
    max_time = occupied_df['datetime'].max().ceil('T')
    
    print(f"  Creating time index: From {min_time} to {max_time} (1-min freq)")
    
    minute_index = pd.date_range(start=min_time, end=max_time, freq='T')
    master_df = pd.DataFrame(minute_index, columns=['datetime'])
    master_df['join_key'] = master_df['datetime']

    # 3. Merge Master File into new index
    master_df = pd.merge(master_df, occupied_df[['join_key', 'occupied_number']], on='join_key', how='left')

    # 4. Load and Merge All Other Sensors
    print("  Loading and merging all sensors...")
    sensor_targets = get_sensor_targets()
    
    for rel_path, time_col, rename_map in sensor_targets:
        file_path = scenario_path / rel_path
        sensor_df = load_and_prep_sensor(file_path, time_col, rename_map)
        
        if sensor_df is not None:
            master_df = pd.merge(master_df, sensor_df, on='join_key', how='left')
            
    # 5. --- Perform Interpolation ---
    print("  Performing interpolation...")
    
    # Set datetime as index for time-based interpolation
    master_df = master_df.set_index('datetime').drop(columns='join_key')

    # Define column groups for interpolation
    ffill_cols = []
    linear_cols = []
    energy_cols = []
    no_fill_cols = []

    for col in master_df.columns:
        if col.startswith(('Switch_State_ac', 'Wind_Speed_ac', 'Setpoint_ac', 'Light_state_', 'Socket_state_')):
            ffill_cols.append(col)
        elif col == 'occupied_number':
            ffill_cols.append(col)
        elif col.startswith('Wireless_switch_'):
            no_fill_cols.append(col)
        elif col.startswith('Energy_Consumption(Wh)'): # <-- Special group
            energy_cols.append(col)
        else:
            # This includes Illuminance, Workstation_temperature
            linear_cols.append(col)

    # Apply Forward Fill (ffill)
    print(f"  - Applying forward fill (ffill) (limit {INTERPOLATE_LIMIT_MINS} min)...")
    master_df[ffill_cols] = master_df[ffill_cols].ffill(limit=INTERPOLATE_LIMIT_MINS)
    
    # Apply Linear Interpolation
    print(f"  - Applying linear interpolation (linear) (limit {INTERPOLATE_LIMIT_MINS} min)...")
    master_df[linear_cols] = master_df[linear_cols].interpolate(method='linear', limit=INTERPOLATE_LIMIT_MINS, limit_area='inside')

    # Apply special Energy logic
    print("  - Applying special energy interpolation (linear then ffill)...")
    # 1. Try linear first (with limit)
    master_df[energy_cols] = master_df[energy_cols].interpolate(method='linear', limit=INTERPOLATE_LIMIT_MINS, limit_area='inside')
    # 2. Fill remaining NaNs with the last known value (no limit)
    master_df[energy_cols] = master_df[energy_cols].ffill() 

    # 'no_fill_cols' are intentionally left with NaNs

    # 6. Save Final File
    master_df = master_df.reset_index() # Move 'datetime' back to a column
    
    # Ensure all original + new columns exist, even if sensor files were missing
    final_column_list = ['datetime', 'occupied_number']
    final_column_list.extend([col for (rp, tc, rm) in sensor_targets for col in rm.values()])
    
    for col in final_column_list:
        if col not in master_df.columns:
            master_df[col] = np.nan
            
    # Reorder columns
    master_df = master_df[final_column_list]
    
    try:
        master_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        print(f"🎉 Success: Merged file saved to {output_file_path}")
        print(f"   - Total Rows: {len(master_df)}, Total Columns: {len(master_df.columns)}")
    except Exception as e:
        print(f"ERROR: Failed to save file {output_file_path.name}: {e}")

# --- 5. Main Execution ---

def main():
    print(f"Starting processing, base path: {BASE_PATH}")
    
    for scenario_name, output_filename in SCENARIOS.items():
        scenario_data_path = BASE_PATH / scenario_name
        output_file_path = OUTPUT_PATH / output_filename
        
        if not scenario_data_path.exists():
            print(f"\nWARNING: Scenario folder not found: {scenario_data_path}. Skipping.")
            continue
            
        process_scenario(scenario_name, scenario_data_path, output_file_path)
        
    print("\n--- All processing complete ---")

if __name__ == "__main__":
    main()