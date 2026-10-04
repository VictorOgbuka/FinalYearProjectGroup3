import os
import time
import math
import threading
import pickle
import logging
import numpy as np
import requests
from scipy.stats import skew, kurtosis
from flask import Flask, request, jsonify, send_from_directory

# Silence Flask request logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- CONFIG ---
BOT_TOKEN = "8659629331:AAEYlZFkz4vvO5uJbBYQdbRgiP4jlxxZwPM"
CHAT_ID = "1601740884"
OFFLINE_AFTER_S = 5.0        
LIVE_RISK_EVERY_S = 0.5      
GRAPH_STEP = 4               
SCRIPT_START_TIME = time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'template')
if not os.path.isdir(TEMPLATE_DIR):
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

LOG_FILE = os.path.join(BASE_DIR, "system_audit_log.csv")
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Time,Event\n")

# --- SHARED STATE ---
lock = threading.RLock()     
ml_threshold = 0.75
event_log = []

def add_to_log(message):
    with lock:
        timestamp = time.strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        # 1. UI Log
        event_log.insert(0, log_entry)
        if len(event_log) > 50:
            event_log.pop()
            
        # 2. VS Code Terminal
        print(log_entry)
        
        # 3. Permanent CSV Audit
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                safe_msg = message.replace('"', '""').replace(',', ';')
                f.write(f"{timestamp},\"{safe_msg}\"\n")
        except Exception:
            pass

ui_state = {
    "status_title": "MONITORING",
    "status_sub": "Patient is stable and moving normally.",
    "state_code": "GREEN",        
    "impact_g": 0.0,              
    "confidence": 0.0,            
    "fall_probability": None,     
    "last_sync": 0.0,             
    "needs_ack": False,
}

device = {"battery": 100, "runtime_sec": 0, "hz": None}
_hz_count, _hz_t0 = 0, time.time()

def update_device_stats(payload):
    """Measures real sampling rate, forces battery to 100, and tracks seconds."""
    global _hz_count, _hz_t0
    now = time.time()
    
    device["battery"] = 100
    device["runtime_sec"] = now - SCRIPT_START_TIME

    for r in payload:
        if r.get('name') == 'accelerometer':
            _hz_count += 1
            
    elapsed = now - _hz_t0
    if elapsed >= 2.0:
        device["hz"] = _hz_count / elapsed
        _hz_count, _hz_t0 = 0, now

def send_telegram_alert(confidence, impact_g):
    try:
        msg = f"🚨 CRITICAL: Fall Confirmed!\nImpact: {impact_g:.2f}g | ML Confidence: {confidence*100:.1f}%"
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage", data={"chat_id": CHAT_ID, "text": msg}, timeout=5)
        add_to_log("Telegram alert delivered.")
    except Exception:
        add_to_log("Telegram alert FAILED to send.")

# --- ML EXTRACTOR ---
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

def svm_confidence():
    feats = scaler.transform([extract_features(np.array(ring_buffer))])
    return float(svm_model.predict_proba(feats)[0][1])

print("🧠 Loading Custom ML Classifier...")
try:
    with open(os.path.join(BASE_DIR, 'classifier.pkl'), 'rb') as f:
        ml_data = pickle.load(f)
        svm_model = ml_data['model']
        scaler = ml_data['scaler']
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading classifier.pkl: {e}")
    exit()

app = Flask(__name__)
ring_buffer = []
current_gyro = [0.0, 0.0, 0.0]
last_alert_time, last_risk_time = 0, 0
impact_triggered = False
samples_since = 0
trigger_svm = 0.0

@app.route('/')
def index():
    return send_from_directory(TEMPLATE_DIR, 'index.html')

def build_series(buf):
    """Normalized graphing data so it renders beautifully in HTML5 Canvas."""
    if len(buf) == 0:
        return {"accel": [], "gyro": [], "pitch": []}
    b = buf[::GRAPH_STEP]
    
    # Scale physics down to a 0.0 to 1.0 ratio for the UI graph
    accel = np.clip(np.sqrt((b[:, :3] ** 2).sum(axis=1)) / 4.0, 0, 1)
    gyro = np.clip(np.sqrt((b[:, 3:] ** 2).sum(axis=1)) / 360.0, 0, 1)
    pitch = np.clip((np.degrees(np.arctan2(-b[:, 0], np.sqrt(b[:, 1]**2 + b[:, 2]**2))) + 90.0) / 180.0, 0, 1)
    
    return {
        "accel": np.round(accel, 3).tolist(),
        "gyro": np.round(gyro, 3).tolist(),
        "pitch": np.round(pitch, 3).tolist()
    }

@app.route('/status')
def status():
    now = time.time()
    with lock:
        buf = np.array(ring_buffer, dtype=float) if ring_buffer else np.empty((0, 6))
        state, log, dev = dict(ui_state), list(event_log), dict(device)
        
    sync = (now - state["last_sync"]) if state["last_sync"] else None
    online = sync is not None and sync < OFFLINE_AFTER_S
    if not online: dev["hz"] = None
    
    return jsonify({
        "state": state, "log": log, "threshold": ml_threshold,
        "sync_time": sync, "online": online, "device": dev,
        "series": build_series(buf),
    })

@app.route('/update_threshold', methods=['POST'])
def update_threshold():
    global ml_threshold
    ml_threshold = min(1.0, max(0.0, float((request.json or {}).get('threshold', 0.75))))
    add_to_log(f"System: ML Threshold adjusted to {ml_threshold*100:.0f}%.")
    return "OK", 200

@app.route('/acknowledge', methods=['POST'])
def acknowledge():
    global impact_triggered, samples_since, trigger_svm
    with lock:
        ui_state.update({'state_code': "GREEN", 'status_title': "MONITORING", 'status_sub': "Patient is stable and moving normally.", 'needs_ack': False, 'fall_probability': None})
        ring_buffer.clear()
        impact_triggered, samples_since, trigger_svm = False, 0, 0.0
        add_to_log("Caregiver acknowledged alarm. System reset to normal monitoring.")
    return "OK", 200

@app.route('/data', methods=['POST'])
def receive_data():
    global last_alert_time, last_risk_time, current_gyro
    global impact_triggered, samples_since, trigger_svm

    data = request.json
    if not data or 'payload' not in data: return "OK", 200

    current_time = time.time()
    with lock:
        ui_state["last_sync"] = current_time
        update_device_stats(data['payload'])

        if ui_state["needs_ack"]: return "OK", 200

        for reading in data['payload']:
            s_name = reading.get('name')
            v = reading.get('values', {})

            if s_name == 'gyroscope':
                current_gyro = [v.get('x', 0) * (180 / math.pi), v.get('y', 0) * (180 / math.pi), v.get('z', 0) * (180 / math.pi)]
            elif s_name == 'accelerometer':
                ax, ay, az = v.get('x', 0) / 9.81, v.get('y', 0) / 9.81, v.get('z', 0) / 9.81
                ring_buffer.append([ax, ay, az, *current_gyro])
                if len(ring_buffer) > 250: ring_buffer.pop(0)

                if len(ring_buffer) == 250:
                    svm_cur = math.sqrt(ax**2 + ay**2 + az**2)

                    if (LIVE_RISK_EVERY_S and not impact_triggered and current_time - last_risk_time >= LIVE_RISK_EVERY_S):
                        ui_state["fall_probability"] = svm_confidence() * 100
                        last_risk_time = current_time

                    if impact_triggered:
                        samples_since += 1
                        if svm_cur > trigger_svm: trigger_svm = svm_cur
                        if samples_since >= 125:
                            conf = svm_confidence()
                            ui_state.update({"impact_g": trigger_svm, "confidence": conf * 100, "fall_probability": conf * 100})
                            
                            if conf > ml_threshold:
                                ui_state.update({"state_code": "RED", "status_title": "CRITICAL: FALL DETECTED", "status_sub": "Patient requires immediate assistance.", "needs_ack": True})
                                add_to_log(f"Fall Confirmed! Confidence: {conf*100:.1f}%, Impact: {trigger_svm:.2f}g")
                                threading.Thread(target=send_telegram_alert, args=(conf, trigger_svm), daemon=True).start()
                            else:
                                ui_state.update({"state_code": "GREEN", "status_title": "MONITORING", "status_sub": "Patient is stable and moving normally."})
                                add_to_log(f"False Alarm Rejected. (Movement: {conf*100:.1f}%)")

                            last_alert_time = current_time
                            impact_triggered, samples_since, trigger_svm = False, 0, 0.0

                    elif (current_time - last_alert_time) > 5.0:
                        recent_min = min(math.sqrt(r[0]**2 + r[1]**2 + r[2]**2) for r in ring_buffer[-50:])
                        if svm_cur > 2.50 and recent_min < 0.40:
                            impact_triggered, trigger_svm = True, svm_cur
                            ui_state.update({"state_code": "YELLOW", "status_title": "PROCESSING..", "status_sub": "Analyzing post-impact aftermath..."})
                            add_to_log("Impact detected, analyzing aftermath.")

    return "OK", 200

if __name__ == '__main__':
    add_to_log("System Armed. Awaiting sensor data...")
    print(f"\n🚀 FallGuard Advanced Edge Server Online (Port 5000), serving UI from: {TEMPLATE_DIR}")
    app.run(host='0.0.0.0', port=5000)