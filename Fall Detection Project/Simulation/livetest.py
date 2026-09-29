import time
import math
import threading
import pickle
import numpy as np
import logging
from scipy.stats import skew, kurtosis
from flask import Flask, request
import requests

# Silence the standard Flask web logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

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
        print("\n⚠️ Failed to send Telegram alert.")

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

# --- INITIALIZE EDGE SERVER ---
app = Flask(__name__)

ring_buffer = [] 
last_alert_time = 0
cooldown_period = 5.0
current_gyro = [0.0, 0.0, 0.0]

# Temporal alignment flags
impact_triggered = False
samples_since_impact = 0
trigger_svm = 0.0

@app.route('/data', methods=['POST'])
def receive_data():
    global ring_buffer, last_alert_time, current_gyro
    global impact_triggered, samples_since_impact, trigger_svm
    
    data = request.json
    if not data or 'payload' not in data:
        return "OK", 200
        
    current_time = time.time()
    
    for reading in data['payload']:
        sensor_name = reading.get('name')
        values = reading.get('values', {})
        
        if sensor_name == 'gyroscope':
            current_gyro = [
                values.get('x', 0) * (180.0 / math.pi),
                values.get('y', 0) * (180.0 / math.pi),
                values.get('z', 0) * (180.0 / math.pi)
            ]
            
        elif sensor_name == 'accelerometer':
            ax = values.get('x', 0) / 9.81
            ay = values.get('y', 0) / 9.81
            az = values.get('z', 0) / 9.81
            
            ring_buffer.append([ax, ay, az, current_gyro[0], current_gyro[1], current_gyro[2]])
            
            if len(ring_buffer) > 250:
                ring_buffer.pop(0)
                
            if len(ring_buffer) == 250:
                svm_current = math.sqrt(ax**2 + ay**2 + az**2)
                
                # STAGE 2: If we are currently recording the aftermath of an impact
                if impact_triggered:
                    samples_since_impact += 1
                    
                    if svm_current > trigger_svm: 
                        trigger_svm = svm_current # Update max impact if it gets higher
                        
                    if samples_since_impact >= 125: # 1.25 seconds of post-impact data collected
                        print(f"⚠️ BUFFER COMPLETE! Max Impact: {trigger_svm:.2f}g. Running ML analysis...")
                        
                        features = extract_features(np.array(ring_buffer))
                        features_scaled = scaler.transform([features])
                        probs = svm_model.predict_proba(features_scaled)[0]
                        confidence = probs[1] 
                        
                        if confidence > 0.65:
                            print(f"🚨 ML CONFIRMED FALL! (Confidence: {confidence*100:.1f}%)")
                            threading.Thread(target=send_telegram_alert, args=(confidence, trigger_svm), daemon=True).start()
                        else:
                            print(f"❌ ML Rejected False Alarm (Confidence: {confidence*100:.1f}%)")
                            
                        # Reset for next event
                        last_alert_time = current_time
                        impact_triggered = False
                        samples_since_impact = 0
                        trigger_svm = 0.0
                        
                # STAGE 1: Normal background monitoring
                elif (current_time - last_alert_time) > cooldown_period:
                    if svm_current > 2.50: 
                        recent_svms = [math.sqrt(row[0]**2 + row[1]**2 + row[2]**2) for row in ring_buffer[-50:]]
                        
                        if min(recent_svms) < 0.40: 
                            print(f"\n⚡ IMPACT DETECTED ({svm_current:.2f}g)! Recording 1.25s of aftermath data...")
                            impact_triggered = True
                            samples_since_impact = 0
                            trigger_svm = svm_current

    return "OK", 200

if __name__ == '__main__':
    print("\n🚀 Starting IoT Edge Server on port 5000...")
    print("Waiting for Sensor Logger data...\n")
    app.run(host='0.0.0.0', port=5000)