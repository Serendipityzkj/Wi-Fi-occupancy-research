import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates

# --- 1. Configuration ---

# Relative path: Assumes the script is next to 'Sensor_data_during_experiment'
BASE_PATH = Path("Sensor_data_during_experiment")

# Select Scenario ('Baseline scenario' or 'OCC scenario')
SCENARIO_NAME = 'Baseline scenario'

# Select Target Date (Format: 'YYYY-MM-DD')
TARGET_DATE = '2025-05-25'

# Interpolation interval (e.g., '1min' for high resolution/smoothness)
INTERPOLATION_INTERVAL = '1min'

# --- 2. Data Processing Function ---

def load_and_process_temp(file_path, target_date_str):
    """
    Loads a temperature sensor file, filters for the target date,
    and performs linear interpolation to create a smooth time series.
    """
    if not file_path.exists():
        print(f"  - Warning: File not found {file_path.name}")
        return None

    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Clean column names
        df.columns = [c.strip() for c in df.columns]
        
        # Convert Time column
        # Using 'mixed' format to handle potential variations
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=False)
        
        # Filter for the target date
        target_date = pd.Timestamp(target_date_str).date()
        df_day = df[df['Time'].dt.date == target_date].copy()
        
        if df_day.empty:
            print(f"  - Warning: No data found for date {target_date_str} in {file_path.name}")
            return None

        # Sort by time
        df_day = df_day.sort_values('Time')
        
        # --- Interpolation Step ---
        # 1. Set Time as index
        df_day = df_day.set_index('Time')
        
        # 2. Resample to a regular interval (e.g., 1 minute) to create a regular grid
        #    We take the mean if multiple readings fall in the same minute.
        df_resampled = df_day.resample(INTERPOLATION_INTERVAL).mean()
        
        # 3. Perform Linear Interpolation to fill missing values (make it smooth)
        #    'time' method interpolates based on the index timestamps
        df_smooth = df_resampled.interpolate(method='time')
        
        return df_smooth

    except Exception as e:
        print(f"  - Error processing {file_path.name}: {e}")
        return None

# --- 3. Plotting Function ---

def plot_all_sensors(sensor_data_dict, date_str, scenario):
    """
    Plots 8 subplots (4x2 grid) for the temperature sensors.
    """
    # Setup figure: 2 rows, 4 columns
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten() # Flatten to easily loop through 0-7
    
    # Set global plot title
    fig.suptitle(f'Workstation Temperature Overview\nScenario: {scenario} | Date: {date_str}', fontsize=16)
    
    # Loop through sensors 1-8
    for i in range(1, 9):
        ax = axes[i-1] # 0-indexed
        sensor_key = f"temp_sensor{i}"
        
        if sensor_key in sensor_data_dict and sensor_data_dict[sensor_key] is not None:
            df = sensor_data_dict[sensor_key]
            col_name = 'Indoor_temperature' # Check your CSV column name if different
            
            if col_name in df.columns:
                # Plot data
                ax.plot(df.index, df[col_name], color='orange', linewidth=2)
                
                # Fill area under curve for better visuals
                ax.fill_between(df.index, df[col_name], min(df[col_name].min(), 20), color='orange', alpha=0.1)
                
                # Stats in title
                avg_temp = df[col_name].mean()
                ax.set_title(f"Sensor {i} (Avg: {avg_temp:.1f}°C)", fontsize=12)
            else:
                ax.text(0.5, 0.5, "Column Not Found", ha='center', va='center')
                ax.set_title(f"Sensor {i}")
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(f"Sensor {i} (Missing)")
            
        # Formatting Axes
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Format X-axis dates (Hour:Minute)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4)) # Tick every 4 hours
    
    # Set common labels
    fig.text(0.5, 0.04, 'Time of Day', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'Temperature (°C)', va='center', rotation='vertical', fontsize=14)
    
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95]) # Adjust layout to fit titles
    plt.show()

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Path to the specific data folder
    # e.g. Original_experimental_data/Baseline scenario/temp_sensor_data
    folder_path = BASE_PATH / SCENARIO_NAME / "temp_sensor_data"
    
    if not folder_path.exists():
        print(f"Error: Data folder not found at {folder_path}")
        print("Please ensure the script is in the same directory as 'Original_experimental_data'.")
    else:
        print(f"Processing Scenario: {SCENARIO_NAME}, Date: {TARGET_DATE}")
        
        # Dictionary to store processed dataframes
        all_sensor_data = {}
        
        # Loop through 1 to 8
        for i in range(1, 9):
            file_name = f"temp_sensor{i}.csv"
            file_path = folder_path / file_name
            
            print(f"  - Reading {file_name}...")
            df_processed = load_and_process_temp(file_path, TARGET_DATE)
            all_sensor_data[f"temp_sensor{i}"] = df_processed
            
        # Plot
        print("Generating plots...")
        plot_all_sensors(all_sensor_data, TARGET_DATE, SCENARIO_NAME)