import pandas as pd
import numpy as np
import os
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Configuration ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Relative paths
DATA_FILE = "Input_data.csv"
MODEL_OUTPUT_PATH = "random_forest_model_best.pkl"

def main():
    print("--- Starting Random Forest Correction Model Training ---")

    # --- 1. Load Data ---
    if not os.path.exists(DATA_FILE):
        print(f"- Error: Data file not found at {DATA_FILE}")
        print("  Please make sure 'Input_data.csv' is in the same directory.")
        return

    try:
        input_data = pd.read_csv(DATA_FILE)
        print(f"- Successfully loaded data from {DATA_FILE}")
        print(f"  Initial data shape: {input_data.shape}")
    except Exception as e:
        print(f"- An unexpected error occurred while loading data: {e}")
        return

    # --- 2. Feature Engineering and Preprocessing ---
    print("- Starting feature engineering...")

    # Define Mappings
    # Map string states to numbers for Features (History States)
    state_to_numeric = {
        "arrival": 0,
        "stay": 1,
        "leave": 2,
        "outside": 3
    }

    # Map for Target (Correction Logic)
    # The RF model predicts: Should we correct this 'outside' prediction to 'IN' (1) or keep it 'OUT' (0)?
    # Real State 'stay'/'arrival' -> 1 (Correction needed: Change 'outside' to 'stay')
    # Real State 'outside'/'leave' -> 0 (No correction needed: Keep 'outside')
    target_mapping = {
        "arrival": 1, 
        "stay": 1,     
        "leave": 0,    
        "outside": 0   
    }

    # Define Feature Columns (History)
    feature_cols = ['State_t-5', 'State_t-4', 'State_t-3', 'State_t-2', 'State_t-1']
    
    # Check if columns exist
    if not all(col in input_data.columns for col in feature_cols + ['State_t']):
        print("- Error: Input data is missing required columns.")
        return

    # Create a copy for X
    X = input_data[feature_cols].copy()
    
    # Apply mapping to feature columns
    for col in feature_cols:
        X[col] = X[col].map(state_to_numeric)
        
    # Apply mapping to target column
    y = input_data['State_t'].map(target_mapping)

    # Handle NaNs (if any non-standard states existed)
    if X.isnull().values.any() or y.isnull().values.any():
        print("- Warning: NaN values found after mapping. Dropping invalid rows.")
        # Combine to drop correctly
        temp_df = pd.concat([X, y.rename('target')], axis=1)
        temp_df = temp_df.dropna()
        X = temp_df[feature_cols]
        y = temp_df['target']

    if X.empty:
        print("- Error: No valid data remaining after preprocessing.")
        return

    print("- Feature engineering complete.")
    print(f"  Features (X) shape: {X.shape}")
    print(f"  Target (y) shape: {y.shape}")

    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print("- Data split into training and testing sets.")

    # --- 4. Grid Search and Training ---
    print("- Defining Random Forest model and parameter grid...")
    
    param_grid = {
        'n_estimators': [50, 100, 150],       # Number of trees
        'max_depth': [5, 10, 15, 20],         # Max depth
        'min_samples_split': [2, 5, 10],      # Min samples to split
        'min_samples_leaf': [1, 2, 4],        # Min samples at leaf
        'class_weight': ['balanced'],         # Handle class imbalance
    }

    rf_model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        verbose=0,
        n_jobs=-1
    )

    print("- Starting training (GridSearchCV)...")
    grid_search.fit(X_train, y_train)
    
    print("  GridSearch Results:")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Best Cross-Validation Score (Accuracy): {grid_search.best_score_:.4f}")

    # --- 5. Evaluation ---
    print("- Evaluating best model on test set...")
    
    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, target_names=['Keep Outside (0)', 'Correct to Stay (1)'])

    print("  Model Evaluation Results (Test Set):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Confusion Matrix: \n{conf_matrix}")
    print(f"  Classification Report: \n{class_report}")

    # --- 6. Plot Confusion Matrix ---
    print("- Generating confusion matrix plot...")
    try:
        class_labels = ['Keep Outside (0)', 'Correct to Stay (1)']
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix - RF Correction Model')
        plt.tight_layout()
        plt.show()
        print("- Plot closed.")
    except Exception as e:
        print(f"- Warning: Could not generate plot: {e}")

    # --- 7. Save Model ---
    try:
        joblib.dump(best_rf_model, MODEL_OUTPUT_PATH)
        print(f"- Best model successfully saved to: {MODEL_OUTPUT_PATH}")
    except Exception as e:
        print(f"- Error saving model: {e}")

    print("--- Process Finished ---")

if __name__ == "__main__":
    main()