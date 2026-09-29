import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# --- 18-FEATURE EXTRACTOR ---
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

def load_custom_data(data_folder):
    X_features = []
    y_labels = []
    
    categories = {'FALLS': 1, 'ADLS': 0}
    
    for category, label in categories.items():
        folder_path = os.path.join(data_folder, category)
        if not os.path.exists(folder_path):
            continue
            
        for file_name in os.listdir(folder_path):
            if not file_name.endswith('.csv'):
                continue
                
            try:
                # Read the 250-sample event window
                df = pd.read_csv(os.path.join(folder_path, file_name), header=None)
                window_data = df.values.astype(float)
                
                if len(window_data) != 250:
                    continue 
                    
                features = extract_features(window_data)
                X_features.append(features)
                y_labels.append(label)
                
            except Exception as e:
                pass
                
    return np.array(X_features), np.array(y_labels)

print("⏳ Loading your custom Samsung Galaxy S20 dataset...")
X, y = load_custom_data("CustomData")
print(f"✅ Loaded {len(X)} total events.")

if len(X) < 10:
    print("❌ Not enough data! Please record more falls and ADLs.")
    exit()

print("\n⏳ Splitting data into training and test sets...")
# Stratify ensures we keep a balanced ratio of Falls and ADLs in the test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("⏳ Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⏳ Training Custom SVM model...")
param_grid = {
    'C': [0.1, 1, 10, 100], 
    'gamma': ['scale', 'auto', 0.1, 0.01], 
    'kernel': ['rbf']
}
# cv=3 because the custom dataset is smaller than Kaggle
svm_model = GridSearchCV(SVC(probability=True), param_grid, cv=3, verbose=0, n_jobs=-1)
svm_model.fit(X_train_scaled, y_train)

print("\n==========================================")
print("--- 🎯 CUSTOM MODEL PERFORMANCE 🎯 ---")
print("==========================================")
y_pred = svm_model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n⏳ Saving your tailored model to classifier.pkl...")
with open('classifier.pkl', 'wb') as f:
    pickle.dump({'model': svm_model.best_estimator_, 'scaler': scaler}, f)
print("✅ SUCCESS! Custom classifier.pkl has overwritten the Kaggle model.")