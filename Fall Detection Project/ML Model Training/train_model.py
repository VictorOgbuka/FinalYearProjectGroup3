import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import skew, kurtosis
from scipy.signal import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# PART 1: THE 18-FEATURE EXTRACTOR
# ==========================================
def extract_features(window_data):
    ax, ay, az = window_data[:, 0], window_data[:, 1], window_data[:, 2]
    gx, gy, gz = window_data[:, 3], window_data[:, 4], window_data[:, 5]
    
    svm = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) * 180 / np.pi
    roll = np.arctan2(ay, az) * 180 / np.pi
    
    return np.array([
        np.mean(svm), np.max(svm), np.min(svm), np.std(svm), skew(svm), kurtosis(svm),
        np.mean(gyro_mag), np.max(gyro_mag), np.std(gyro_mag), (np.max(svm) - np.min(svm)),
        (np.argmax(svm) / 100.0), np.mean(svm[-50:]), np.std(svm[-50:]),
        np.mean(pitch), (pitch[-1] - pitch[0]), np.max(roll),
        np.where(np.diff(np.sign(svm - np.mean(svm))))[0].size, np.sum(svm**2)
    ])

# ==========================================
# PART 2: DATA LOADER & PREPROCESSOR
# ==========================================
def load_sisfall_data(data_folder):
    print("⏳ Loading and calibrating SisFall data to standard units ('g' and 'deg/s')...")
    
    X_features = []
    y_labels = []
    
    for subject_folder in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject_folder)
        if not os.path.isdir(subject_path):
            continue
            
        for file_name in os.listdir(subject_path):
            if not file_name.endswith('.txt'):
                continue
                
            is_fall = 1 if file_name.startswith('F') else 0
            
            try:
                # Read the raw CSV file
                df = pd.read_csv(os.path.join(subject_path, file_name), sep=',', header=None, usecols=[0,1,2,3,4,5], on_bad_lines='skip')
                raw_data = df.values.astype(float)
                
                if len(raw_data) < 500: 
                    continue
                    
                # --- UNIVERSAL CALIBRATION ---
                # Convert SisFall raw integers to universal 'g' and 'deg/s'
                raw_data[:, 0:3] = raw_data[:, 0:3] / 256.0   # Accelerometer to 'g'
                raw_data[:, 3:6] = raw_data[:, 3:6] / 14.375  # Gyroscope to 'deg/s'
                
                # Downsample 200Hz to 100Hz
                new_length = len(raw_data) // 2
                resampled_data = resample(raw_data, new_length)
                
                # Windowing around impact peak
                ax, ay, az = resampled_data[:, 0], resampled_data[:, 1], resampled_data[:, 2]
                svm = np.sqrt(ax**2 + ay**2 + az**2)
                peak_index = np.argmax(svm)
                
                start_idx = peak_index - 125
                end_idx = peak_index + 125
                
                if start_idx < 0 or end_idx >= len(resampled_data):
                    continue
                    
                window = resampled_data[start_idx:end_idx]
                features = extract_features(window)
                
                X_features.append(features)
                y_labels.append(is_fall)
                
            except Exception as e:
                pass
                
    return np.array(X_features), np.array(y_labels)

X, y = load_sisfall_data("SisFall_Data")
print(f"✅ Data processing complete! Valid samples: {len(X)}")

# ==========================================
# PART 3: TRAINING THE SVM & SAVING THE MODEL
# ==========================================
print("\n⏳ Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("⏳ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⏳ Training the SVM model (Grid Search for best hyperparameters)...")
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto', 0.1, 0.01], 
    'kernel': ['rbf']
}

svm_model = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=1, n_jobs=-1)
svm_model.fit(X_train_scaled, y_train)

print(f"\n✅ Best parameters found: {svm_model.best_params_}")

print("\n⏳ Evaluating the model on the test data...")
y_pred = svm_model.predict(X_test_scaled)

print("\n==========================================")
print("--- 🎯 MODEL PERFORMANCE RESULTS 🎯 ---")
print("==========================================")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n⏳ Saving the model to classifier.pkl...")
with open('classifier.pkl', 'wb') as f:
    pickle.dump({'model': svm_model.best_estimator_, 'scaler': scaler}, f)

print("✅ SUCCESS! classifier.pkl has been created.")