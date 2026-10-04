import csv
import time

# --- STATE VARIABLES ---
last_alert_time = -10.0  # Set in the past so it triggers immediately on the first fall
cooldown_period = 5.0    # Wait 5 seconds after a fall before scanning again

with open('Raw Data.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader) # Skip the header row
    
    print("Starting Fall Detection Simulation (V2)...\n")
    
    for row in reader:
        try:
            timestamp = float(row[0])
            
            # OPTIMIZATION: We no longer calculate X, Y, and Z manually.
            # Column E (index 4) is already Absolute Acceleration!
            sv = float(row[4]) 
            
            # Only run the detection logic IF we are not in a cooldown period
            if (timestamp - last_alert_time) > cooldown_period:
                
                # Check for the hard impact
                if sv > 25.0:
                    print(f"Time: {timestamp:.1f}s | SV: {sv:.2f} m/s²")
                    print("🚨 CRITICAL: FALL IMPACT DETECTED! 🚨")
                    print("Triggering IoT Alert System...")
                    print(f"Cooling down for {cooldown_period} seconds...\n")
                    
                    # Update the last alert time to right now
                    last_alert_time = timestamp
                    
                    time.sleep(2) # Simulate network delay for sending SMS
            
            # Speeding up the loop delay because your CSV has thousands of rows per second
            time.sleep(0.001) 
            
        except (ValueError, IndexError):
            continue