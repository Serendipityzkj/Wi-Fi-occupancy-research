import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# --- 1. Configuration ---

# Relative path: Assumes the script is next to 'Sensor_data_during_experiment'
BASE_PATH = Path("Sensor_data_during_experiment")

# Select Scenario ('Baseline scenario' or 'OCC scenario')
SCENARIO_NAME = 'Baseline scenario'

# Select Target Date (Format: 'YYYY-MM-DD')
# Ensure this date exists in your CSV files
TARGET_DATE = '2025-05-25'

# --- 2. Core Processing Functions ---

def load_and_interpolate_ac(file_path, target_date_str):
    """
    Loads a single AC accumulated energy file and interpolates values 
    at exact hourly intervals for the target date (00:00 to next day 00:00).
    """
    if not file_path.exists():
        print(f"Error: File not found {file_path}")
        return None

    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Clean column names (remove potential spaces)
        df.columns = [c.strip() for c in df.columns]
        
        # Convert Time column
        # format='mixed' allows pandas to infer format (e.g., 2025/5/25 0:00)
        df['Time'] = pd.to_datetime(df['Time'], format='mixed', dayfirst=False)
        
        # Sort by time (required for interpolation)
        df = df.sort_values('Time').drop_duplicates('Time')
        
        # Identify the value column (Accumulated Energy)
        # Looks for columns containing 'Accumulated' or 'Energy'
        val_col = [c for c in df.columns if 'Accumulated' in c or 'Energy' in c][0]
        
        # --- Create Target Timeline ---
        # From Target 00:00 to Next Day 00:00 (25 points total)
        target_start = pd.Timestamp(target_date_str)
        target_end = target_start + pd.Timedelta(days=1)
        target_timestamps = pd.date_range(start=target_start, end=target_end, freq='H')
        
        # --- Linear Interpolation ---
        # Convert timestamps to float (seconds) for mathematical interpolation
        x_new = target_timestamps.astype(np.int64) // 10**9
        xp = df['Time'].astype(np.int64) // 10**9
        fp = df[val_col].values
        
        # Perform interpolation
        interpolated_values = np.interp(x_new, xp, fp)
        
        return interpolated_values

    except Exception as e:
        print(f"Error processing file {file_path.name}: {e}")
        return None

def calculate_daily_hourly_consumption(base_path, scenario, date_str):
    """
    Calculates the sum of hourly energy consumption for AC1, AC2, and AC3.
    """
    folder_path = base_path / scenario / "ac_sensor_data"
    
    # Store the interpolated accumulated values (25 points) for each AC
    ac_accumulated_results = []
    
    print(f"Processing Scenario: {scenario}, Date: {date_str}")
    
    for i in range(1, 4):
        file_name = f"ac{i}_accumulated_energy_consumption.csv"
        file_path = folder_path / file_name
        
        print(f"  - Reading {file_name} ...")
        
        interp_vals = load_and_interpolate_ac(file_path, date_str)
        
        if interp_vals is None:
            print(f"    WARNING: Could not get data for AC{i}. Assuming 0 usage.")
            # Append array of zeros if file fails
            ac_accumulated_results.append(np.zeros(25)) 
        else:
            ac_accumulated_results.append(interp_vals)
            
    # --- Calculate Hourly Increment ---
    # Sum the accumulated curves of all 3 ACs
    total_accumulated = np.sum(ac_accumulated_results, axis=0)
    
    # Calculate the difference between hours (np.diff)
    # Result will have 24 elements (Hour 0 to Hour 23)
    hourly_consumption = np.diff(total_accumulated)
    
    return hourly_consumption

# --- 3. Plotting Function ---

def plot_energy(hourly_data, date_str, scenario):
    """
    Plots a bar chart of the hourly energy consumption.
    """
    hours = range(24) # 0, 1, ..., 23
    
    # Set figure size
    plt.figure(figsize=(12, 6))
    
    # Create bar chart
    bars = plt.bar(hours, hourly_data, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom', fontsize=9)
    
    # Labels and Title
    plt.xlabel('Time of Day (Hour)', fontsize=12)
    plt.ylabel('Energy Consumption (Wh)', fontsize=12)
    plt.title(f'Hourly Energy Consumption \nScenario: {scenario} | Date: {date_str}', fontsize=14)
    
    # X-axis ticks (0:00, 1:00, ...)
    plt.xticks(hours, [f"{h}:00" for h in hours], rotation=45)
    
    # Grid and Layout
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Display plot
    plt.show()

# --- 4. Main Execution ---

if __name__ == "__main__":
    # Check if base path exists
    if not BASE_PATH.exists():
        print(f"CRITICAL ERROR: The folder '{BASE_PATH}' was not found in the current directory.")
        print("Please ensure this script is located in the same folder as 'Original_experimental_data'.")
    else:
        # 1. Calculate
        hourly_usage = calculate_daily_hourly_consumption(BASE_PATH, SCENARIO_NAME, TARGET_DATE)
        
        # 2. Check results
        if np.all(hourly_usage == 0):
            print("\nResult Notice: Calculated energy is all 0.")
            print("Possible reasons: ")
            print("1. Data for this date does not exist in the CSVs.")
            print("2. Timestamps in CSV do not match the target date.")
            print("3. Accumulated energy did not change (ACs were off).")
        else:
            print(f"\nCalculation Complete. Total Daily Energy: {np.sum(hourly_usage):.2f} Wh")
            
            # 3. Plot
            plot_energy(hourly_usage, TARGET_DATE, SCENARIO_NAME)