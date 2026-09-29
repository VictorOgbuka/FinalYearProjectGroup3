import requests
import time
import math
import threading
import pickle
import numpy as np
from scipy.stats import skew, kurtosis

# --- TELEGRAM SETUP ---
BOT_TOKEN = "8659629331:AAEYlZFkz4vvO5uJbBYQdbRgiP4jlxxZwPM"
CHAT_ID = "1601740884"

def send_telegram_alert(confidence, impact_g):
    try:
        message = f"🚨 CRITICAL: ML Confirmed Fall! Patient requires immediate assistance.\n(Impact: {impact_g:.2f}g | ML Confidence: {confidence*100:.1f}%)"
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
        requests.get(url, timeout=5) 
        print(f"\n📲 Telegram alert sent successfully! (Confidence: {confidence*100:.1f}%)")
    except Exception as e:
        print("\n⚠️ Failed to send Telegram alert (Network Timeout).")

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

# Helper function to handle 360-degree wrap-around when calculating rotational velocity
def get_angle_diff(current, previous):
    diff = current - previous
    return (diff + 180) % 360 - 180

# --- LOAD THE ML MODEL ---
print("🧠 Loading ML Classifier...")
try:
    with open('classifier.pkl', 'rb') as f:
        ml_data = pickle.load(f)
        svm_model = ml_data['model']
        scaler = ml_data['scaler']
    print("✅ Model loaded successfully!")
except Exception as e:
    print("❌ Error loading classifier.pkl. Make sure it is in the same folder.")
    exit()

# --- SENSOR SETUP ---
URL = "http://10.217.39.132:8080/get?accX&accY&accZ"

print("📡 Connecting to Phone Sensor...")
print("Starting Hybrid Fall Detection (Synthesized Gyro + ML)...")

ring_buffer = [] 
last_alert_time = 0
cooldown_period = 5.0

last_pitch = None
last_roll = None

while True:
    try:
        response = requests.get(URL)
        data = response.json()
        
        x_array = data['buffer']['accX']['buffer']
        y_array = data['buffer']['accY']['buffer']
        z_array = data['buffer']['accZ']['buffer']
        
        current_time = time.time()
        
        for i in range(len(x_array)):
            ax = x_array[i] / 9.81
            ay = y_array[i] / 9.81
            az = z_array[i] / 9.81
            
            # Calculate current postural angles
            current_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay**2 + az**2)))
            current_roll = math.degrees(math.atan2(ay, az))
            
            # SYNTHESIZE GYROSCOPE DATA (Degrees per second)
            # Assuming 100Hz sample rate (dt = 0.01s)
            if last_pitch is not None:
                gx = get_angle_diff(current_roll, last_roll) / 0.01
                gy = get_angle_diff(current_pitch, last_pitch) / 0.01
                gz = 0.0 # Yaw cannot be derived from gravity, but X and Y provide enough rotational energy
            else:
                gx, gy, gz = 0.0, 0.0, 0.0
                
            last_pitch = current_pitch
            last_roll = current_roll
            
            ring_buffer.append([ax, ay, az, gx, gy, gz])
            
            if len(ring_buffer) > 250:
                ring_buffer.pop(0)
                
            if (current_time - last_alert_time) > cooldown_period and len(ring_buffer) == 250:
                svm_current = math.sqrt(ax**2 + ay**2 + az**2)
                
                # STAGE 1: Threshold Event Trigger (Impact > 2.50 g)
                if svm_current > 2.50: 
                    recent_svms = [math.sqrt(row[0]**2 + row[1]**2 + row[2]**2) for row in ring_buffer[-50:]]
                    
                    if min(recent_svms) < 0.40: 
                        print(f"\n⚠️ THRESHOLD TRIGGERED! Impact: {svm_current:.2f}g. Running ML analysis...")
                        
                        # STAGE 2: ML Classification
                        features = extract_features(np.array(ring_buffer))
                        features_scaled = scaler.transform([features])
                        probs = svm_model.predict_proba(features_scaled)[0]
                        confidence = probs[1] 
                        
                        # Filter out non-fall activities
                        if confidence > 0.75:
                            print(f"🚨 ML CONFIRMED FALL! (Confidence: {confidence*100:.1f}%)")
                            threading.Thread(target=send_telegram_alert, args=(confidence, svm_current), daemon=True).start()
                            last_alert_time = current_time
                            break 
                        else:
                            print(f"❌ ML Rejected False Alarm (Confidence: {confidence*100:.1f}%)")
                            last_alert_time = current_time 
                            break
                            
    except Exception as e:
        print(f"⚠️ System Error: {e}")
        time.sleep(1.0) # Slows down the error spam
        
    time.sleep(0.1)