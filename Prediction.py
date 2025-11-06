import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import sys
from collections import deque

# --- 1. Configuration and Paths ---

# Set TensorFlow log level to suppress unnecessary warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Define relative paths
MODEL_TRAINING_FOLDER = "Model training"
CNN_MODEL_PATH = os.path.join(MODEL_TRAINING_FOLDER, "CNN_model_output", "model")
GBC_MODEL_PATH = os.path.join(MODEL_TRAINING_FOLDER, "gradient_boosting_model_best.pkl")
RF_MODEL_PATH = os.path.join(MODEL_TRAINING_FOLDER, "random_forest_model_best.pkl")

INPUT_FILE = "Testing_samples.xlsx"

# --- Model Specifics (from our training scripts) ---

# CNN Model
CNN_WINDOW_SIZE = 6
CNN_FEATURE_COLUMNS = [f'RSSI{i}' for i in range(1, 11)]

# GBC Model
GBC_WINDOW_SIZE = 7  # GBC uses 7 outputs from the CNN

# RF Model
RF_PAST_STATES = 5

# Demo Configuration
DEMO_STEPS = 30 # Number of consecutive states to output

# --- Mappings ---

# The final output labels for the GBC model
OUTPUT_LABELS = ['arrival', 'stay', 'leave', 'outside']

# Map GBC string outputs to the numeric states the RF model was trained on
RF_INPUT_STATE_MAP = {
    'arrival': 0,
    'stay': 1,
    'leave': 2,
    'outside': 3
}

# --- 2. Model Loading Function ---

def load_all_models():
    """
    Attempts to load all three trained models from their respective paths.
    """
    print("--- Loading All Trained Models ---")
    
    cnn_model = None
    try:
        cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
        print(f"✅ Successfully loaded CNN model from: {CNN_MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load CNN model: {e}")

    gbc_model = None
    try:
        gbc_model = joblib.load(GBC_MODEL_PATH)
        print(f"✅ Successfully loaded GBC model from: {GBC_MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load GBC model: {e}")

    rf_model = None
    try:
        rf_model = joblib.load(RF_MODEL_PATH)
        print(f"✅ Successfully loaded RF model from: {RF_MODEL_PATH}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load RF model: {e}")
        
    print("--------------------------------------\n")
    return cnn_model, gbc_model, rf_model

# --- 3. Data Loading Function ---

def load_test_data(file_path, required_cols):
    """
    Loads the testing samples Excel file.
    """
    print(f"--- Loading Test Data ---")
    try:
        df = pd.read_excel(file_path)
        print(f"✅ Successfully loaded test file: {file_path}")
    except FileNotFoundError:
        print(f"❌ ERROR: Test file not found at: {file_path}")
        print(f"Please make sure '{INPUT_FILE}' is in the same directory.")
        return None
    except Exception as e:
        print(f"❌ ERROR: Failed to read Excel file: {e}")
        return None

    # Verify required columns
    if not all(col in df.columns for col in required_cols):
        print(f"❌ ERROR: The test file is missing required RSSI columns.")
        print(f"   Expected: {required_cols}")
        return None
        
    # Verify data length
    # To get N demo steps, we need N-1 + (GBC_WINDOW + CNN_WINDOW - 1) rows
    required_rows = (CNN_WINDOW_SIZE + GBC_WINDOW_SIZE - 1) + DEMO_STEPS - 1
    if len(df) < required_rows:
        print(f"❌ ERROR: Data is too short for a {DEMO_STEPS}-step demo.")
        print(f"   Need at least {required_rows} rows, but found {len(df)}.")
        return None
        
    print(f"   - Found {len(df)} total samples (enough for demo).")
    print("----------------------------\n")
    return df

# --- 4. Prediction Demonstration ---

def run_prediction_demo(cnn_model, gbc_model, rf_model, data_df):
    """
    Runs the continuous, pipelined prediction demo.
    """
    print("--- Starting Pipelined Prediction Demo ---")
    
    # This deque will store the last 5 GBC output labels for the RF model
    state_history = deque(maxlen=RF_PAST_STATES)
    
    # Loop to generate the number of steps you requested
    for i in range(DEMO_STEPS):
        
        # --- 1. Get CNN-GBC Prediction (Base) ---
        
        # Store the 7 CNN outputs (logits)
        cnn_outputs = []
        
        # This inner loop runs the CNN 7 times
        for j in range(GBC_WINDOW_SIZE):
            
            # 1a. Get the 6-row RSSI window
            rssi_start_row = i + j
            rssi_end_row = rssi_start_row + CNN_WINDOW_SIZE
            window_df = data_df.iloc[rssi_start_row : rssi_end_row]
            window_features = window_df[CNN_FEATURE_COLUMNS].to_numpy()
            
            # 1b. Preprocessing (Normalize, Reshape)
            normalized_window = (window_features + 100.0) / 100.0
            reshaped_window = normalized_window.reshape(
                1, CNN_WINDOW_SIZE, len(CNN_FEATURE_COLUMNS), 1
            )
            
            # 1c. Get CNN prediction (logits) and store it
            cnn_prediction_logits = cnn_model.predict(reshaped_window, verbose=0)
            cnn_outputs.append(cnn_prediction_logits.flatten())

        # 1d. Create the GBC input
        # Flatten the 7x4 matrix into a 1x28 vector
        gbc_input = np.array(cnn_outputs).flatten().reshape(1, -1)
        
        # 1e. Make GBC prediction
        gbc_prediction_index = gbc_model.predict(gbc_input)[0]
        
        # This is the "CNN-GBC" prediction label
        cnn_gbc_label = OUTPUT_LABELS[gbc_prediction_index]
        
        # --- 2. Get CNN-GBC-RF Prediction (Pipelined) ---
        
        # Start by assuming the RF model makes no change
        final_output_label = cnn_gbc_label
        reason = ""
        
        if len(state_history) < RF_PAST_STATES:
            # We don't have enough history to run the RF model
            reason = f"(RF disabled: waiting for {RF_PAST_STATES} past states, have {len(state_history)})"
            
        elif cnn_gbc_label == 'outside':
            # We have enough history AND the trigger state
            
            # 2a. Get the 5 past states and map them to numbers
            rf_input_numeric = [RF_INPUT_STATE_MAP[state] for state in state_history]
            rf_input_array = np.array(rf_input_numeric).reshape(1, -1)
            
            # 2b. Make RF prediction
            rf_prediction = rf_model.predict(rf_input_array)[0]
            
            # 2c. Apply correction logic
            if rf_prediction == 1:
                final_output_label = 'stay' # Correct 'outside' to 'stay'
                reason = f"(RF CORRECTION: 'outside' changed to 'stay' based on history)"
            else:
                reason = f"(RF validation: RF model agreed with 'outside')"
        else:
            # The state is not 'outside', so RF is not triggered
            reason = f"(RF not triggered: state is '{cnn_gbc_label}')"

        # --- 3. Print Results ---
        print(f"\n--- Step {i+1} / {DEMO_STEPS} (using RSSI rows {i} to {i + CNN_WINDOW_SIZE + GBC_WINDOW_SIZE - 2}) ---")
        print(f"   CNN-GBC Prediction:    {cnn_gbc_label.upper()}")
        print(f"   CNN-GBC-RF Prediction: {final_output_label.upper()} {reason}")

        # --- 4. Update History ---
        # Add the new CNN-GBC state to the history
        state_history.append(cnn_gbc_label)

        # Pause for a demo effect
        try:
            time.sleep(1.5)
        except KeyboardInterrupt:
            print("\nDemo stopped by user.")
            return

    print("\n\n--- Prediction Demo Finished ---")

# --- 5. Main Execution ---

def main():
    # Load all models
    cnn_model, gbc_model, rf_model = load_all_models()
    
    # Load the input data
    test_data = load_test_data(INPUT_FILE, CNN_FEATURE_COLUMNS)
    
    # Check if we can proceed
    if cnn_model is None or gbc_model is None or rf_model is None or test_data is None:
        print("ERROR: Cannot start demo. Missing critical model (CNN, GBC, or RF) or data.")
        print("Please check file paths and errors above.")
        sys.exit(1) # Exit the script with an error
        
    # Inform the user about the pipeline
    print("NOTE: This demo will run a complex CNN -> GBC -> RF pipeline.")
    print(f"   1. 'CNN-GBC': CNN runs {GBC_WINDOW_SIZE} times, its (7,4) output is flattened")
    print(f"      and fed to GBC for a final state prediction.")
    print(f"   2. 'CNN-GBC-RF': RF model corrects the 'outside' prediction")
    print(f"      based on the last {RF_PAST_STATES} states.")
    print("-------------------------------------------------------------------\n")
    
    # Run the demo
    run_prediction_demo(cnn_model, gbc_model, rf_model, test_data)

if __name__ == "__main__":
    main()