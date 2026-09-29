import time
import math
import os
import pandas as pd
import numpy as np
import logging
from flask import Flask, request

# Silence Flask logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# ==========================================
# 🛑 CHANGE THIS BEFORE RECORDING 🛑
# Set to "FALLS" when doing actual drops.
# Set to "ADLS" when doing jumps, running, or sitting heavily.
CURRENT_LABEL = "ADLS" 
# ==========================================

# Ensure directories exist
os.makedirs(f"CustomData/{CURRENT_LABEL}", exist_ok=True)

app = Flask(__name__)

ring_buffer = [] 
last_alert_time = 0
cooldown_period = 3.0
current_gyro = [0.0, 0.0, 0.0]

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
                
                if impact_triggered:
                    samples_since_impact += 1
                    if svm_current > trigger_svm: trigger_svm = svm_current
                        
                    if samples_since_impact >= 125:
                        print(f"✅ Captured {CURRENT_LABEL} event! Max Impact: {trigger_svm:.2f}g.")
                        
                        # Save the 250-sample window to a CSV
                        df = pd.DataFrame(ring_buffer, columns=['ax','ay','az','gx','gy','gz'])
                        filename = f"CustomData/{CURRENT_LABEL}/{int(time.time())}.csv"
                        df.to_csv(filename, index=False, header=False)
                        print(f"📁 Saved to {filename}")
                        
                        last_alert_time = current_time
                        impact_triggered = False
                        samples_since_impact = 0
                        trigger_svm = 0.0
                        
                elif (current_time - last_alert_time) > cooldown_period:
                    if svm_current > 2.50: 
                        recent_svms = [math.sqrt(row[0]**2 + row[1]**2 + row[2]**2) for row in ring_buffer[-50:]]
                        if min(recent_svms) < 0.40: 
                            print(f"\n⚡ {CURRENT_LABEL} IMPACT DETECTED! Recording aftermath...")
                            impact_triggered = True
                            samples_since_impact = 0
                            trigger_svm = svm_current

    return "OK", 200

if __name__ == '__main__':
    print(f"\n🚀 Data Collector Running (Target: {CURRENT_LABEL})")
    app.run(host='0.0.0.0', port=5000)