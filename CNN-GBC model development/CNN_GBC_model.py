import pandas as pd
import numpy as np
import os
import shutil
import warnings
import tensorflow as tf
import joblib
import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- Configuration ---

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')

# Define Paths (Relative)
INPUT_FILE = 'Input_data.csv'
SAVED_MODELS_DIR = "Saved_models"
TMP_DIR = "tmp"
CNN_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, "CNN_model")
GBC_SAVE_PATH = os.path.join(SAVED_MODELS_DIR, "gradient_boosting_model_best.pkl")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Global Constants ---
LABELS = ['enter', 'leave', 'stay', 'out']
FEATURE_COLUMNS = [f'RSSI{i}' for i in range(1, 11)]
TARGET_COLUMN = 'state'

# CNN Parameters
CNN_WINDOW_SIZE = 6
NN_CONN2D_1 = 64
NN_CONN2D_2 = 128
NN_NH_1 = 256
LEARNING_RATE = 0.0001
BATCH_SIZE = 100
SAVE_STEP_EPOCHS = 3000 
DISPLAY_STEP = 100

# GBC Parameters
GBC_WINDOW_SIZE = 7

# --- Model Definition ---

class MyCNNModel(Model):
    def __init__(self):
        super(MyCNNModel, self).__init__()
        self.conv1 = Conv2D(NN_CONN2D_1, 3, activation='relu', padding='same')
        self.pool1 = MaxPooling2D(2)
        self.conv2 = Conv2D(NN_CONN2D_2, 2, activation='relu')
        self.pool2 = MaxPooling2D(1)
        self.flatten = Flatten()
        self.d1 = Dense(NN_NH_1, activation='relu')
        self.dropout = Dropout(0.5)
        self.d2 = Dense(len(LABELS))

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.d1(x)
        if training:
            x = self.dropout(x, training=training)
        return self.d2(x)

# --- Helper Functions ---

def create_directories():
    """Create necessary directories if they don't exist."""
    if not os.path.exists(SAVED_MODELS_DIR):
        os.makedirs(SAVED_MODELS_DIR)
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

def load_and_process_cnn_data(csv_path):
    """
    Loads data from CSV and creates sliding windows for CNN input.
    """
    if not os.path.exists(csv_path):
        print(f"- Error: File {csv_path} not found.")
        return None, None

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"- Error reading CSV: {e}")
        return None, None
    
    if len(df) < CNN_WINDOW_SIZE:
        print("- Error: Not enough data.")
        return None, None
        
    all_X = []
    all_y = []
    
    # Apply sliding window
    for i in range(len(df) - CNN_WINDOW_SIZE + 1):
        label_value = df.iloc[i + CNN_WINDOW_SIZE - 1][TARGET_COLUMN]
        
        if pd.isna(label_value) or label_value not in LABELS:
            continue
            
        window_features = df.iloc[i : i + CNN_WINDOW_SIZE][FEATURE_COLUMNS].to_numpy()
        
        # Normalize
        normalized_window = (window_features + 100.0) / 100.0
        
        # Reshape for CNN
        reshaped_window = normalized_window.reshape(CNN_WINDOW_SIZE, 10, 1)
        
        # One-hot encode label
        label = np.zeros(len(LABELS))
        label[LABELS.index(label_value)] = 1
        
        all_X.append(reshaped_window)
        all_y.append(label)
        
    return np.array(all_X), np.array(all_y)

def train_cnn_model(X_train, y_train, X_all_shape):
    """
    Initializes and trains the CNN model.
    """
    print(f"- Starting CNN Training for {SAVE_STEP_EPOCHS} epochs...")
    
    cnn_model = MyCNNModel()
    # Build model to fix input shape
    cnn_model.build(input_shape=(None, CNN_WINDOW_SIZE, 10, 1))
    
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = cnn_model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, cnn_model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, cnn_model.trainable_variables))
        train_loss(loss)
        train_accuracy(labels, predictions)

    total_batch = int(len(X_train) / BATCH_SIZE)
    start_time = time.time()

    for epoch in range(SAVE_STEP_EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for batch in range(total_batch):
            start_idx = batch * BATCH_SIZE
            end_idx = (batch + 1) * BATCH_SIZE
            batch_xs = X_shuffled[start_idx:end_idx]
            batch_ys = y_shuffled[start_idx:end_idx]
            
            train_step(batch_xs, batch_ys)
            
        if (epoch + 1) % DISPLAY_STEP == 0:
            print(f"  Epoch {epoch+1}, Loss: {train_loss.result():.4f}, Accuracy: {train_accuracy.result()*100:.2f}%")

    print(f"- CNN Training finished in {time.time() - start_time:.2f} seconds.")
    
    # Save Model
    cnn_model.save(CNN_SAVE_PATH, save_format="tf")
    print(f"- CNN Model saved to: {CNN_SAVE_PATH}")
    
    return cnn_model

def generate_gbc_data(cnn_model, X_cnn_all, y_cnn_all):
    """
    Generates input features for the GBC model using the trained CNN.
    """
    print("- Generating input features for GBC model using trained CNN...")
    
    # Get predictions for the entire dataset
    cnn_predictions_logits = cnn_model.predict(X_cnn_all, verbose=0)
    
    gbc_features = []
    gbc_labels = []
    
    y_indices = np.argmax(y_cnn_all, axis=1)
    
    # Create sliding windows of CNN outputs
    for i in range(len(cnn_predictions_logits) - GBC_WINDOW_SIZE + 1):
        window_preds = cnn_predictions_logits[i : i + GBC_WINDOW_SIZE].flatten()
        target_label = y_indices[i + GBC_WINDOW_SIZE - 1]
        
        gbc_features.append(window_preds)
        gbc_labels.append(target_label)
        
    return np.array(gbc_features), np.array(gbc_labels)

def train_gbc_model(X_train, y_train):
    """
    Trains the GBC model using GridSearchCV.
    """
    print("- Starting GBC Training with Grid Search...")
    
    gradient_boosting = GradientBoostingClassifier()
    
    param_grid = {
        'n_estimators': [70, 80, 85, 86, 90],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.45],
        'max_depth': [2, 3, 4, 5, 6]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(gradient_boosting, param_grid, cv=5, n_jobs=-1)
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    
    print(f"- GBC Training finished in {time.time() - start_time:.2f} seconds.")
    print(f"  Best parameters: {grid_search.best_params_}")
    
    # Save Model
    joblib.dump(grid_search.best_estimator_, GBC_SAVE_PATH)
    print(f"- GBC Model saved to: {GBC_SAVE_PATH}")
    
    return grid_search

# --- Main Execution ---

def main():
    print("--- Starting Sequential Training Pipeline (CNN -> GBC) ---")
    
    create_directories()
    
    # 1. Load CNN Data
    print("- Loading and processing data for CNN...")
    X_cnn_all, y_cnn_all = load_and_process_cnn_data(INPUT_FILE)
    
    if X_cnn_all is None:
        return

    # Split for CNN
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(
        X_cnn_all, y_cnn_all, test_size=0.2, random_state=42
    )
    print(f"- CNN Data Loaded. Total samples: {len(X_cnn_all)}")

    # 2. Train CNN
    cnn_model = train_cnn_model(X_train_cnn, y_train_cnn, X_cnn_all.shape)

    # 3. Generate GBC Data
    X_gbc, y_gbc = generate_gbc_data(cnn_model, X_cnn_all, y_cnn_all)
    print(f"- GBC Features generated. Shape: {X_gbc.shape}")

    # Split for GBC
    X_train_gbc, X_test_gbc, y_train_gbc, y_test_gbc = train_test_split(
        X_gbc, y_gbc, test_size=0.2, random_state=42, stratify=y_gbc
    )

    # 4. Train GBC
    grid_search_gbc = train_gbc_model(X_train_gbc, y_train_gbc)

    # 5. Evaluate GBC
    print("- Evaluating GBC Model on Test Set...")
    y_pred_gbc = grid_search_gbc.predict(X_test_gbc)
    print("  Gradient Boosting Classification Report:")
    print(classification_report(y_test_gbc, y_pred_gbc, target_names=LABELS))

    print("- All training tasks completed successfully.")

if __name__ == "__main__":
    main()