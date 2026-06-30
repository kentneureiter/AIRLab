#reads inference data from serial port instead of through OpenMV ide and sends to CSV file after
import serial
import csv
import time
from datetime import datetime

PORT = "/dev/cu.usbmodem395B355A30312"
BAUD_RATE = 115200
NUM_SAMPLES = 100
OUTPUT_FILE = "inference_log.csv"


samples = []

print(f"Opening port {PORT}...")
ser = serial.Serial(PORT, BAUD_RATE, timeout=2)
time.sleep(2)  # wait for connection to stabilize

print(f"Collecting {NUM_SAMPLES} samples...")
while len(samples) < NUM_SAMPLES:
    line = ser.readline().decode("utf-8").strip()
    if line:
        try:
            value = float(line)
            samples.append(value)
            print(f"Sample {len(samples)}: {value} ms")
        except ValueError:
            pass  # skip non-numeric lines like debug prints
ser.close()

minVal = min(samples)
maxVal = max(samples)
averageVal = sum(samples)/len(samples)
print(minVal)
print(maxVal)
print(averageVal)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    
    # header row
    writer.writerow(["Sample #", "Inference Time (ms)", "Timestamp"])
    
    # one row per sample
    for i, val in enumerate(samples):
        writer.writerow([i+1, round(val, 3), datetime.now().strftime("%H:%M:%S")])
    
    # empty row then summary
    writer.writerow([])
    writer.writerow(["Min (ms)", round(minVal, 3)])
    writer.writerow(["Max (ms)", round(maxVal, 3)])
    writer.writerow(["Average (ms)", round(averageVal, 3)])
    writer.writerow(["Samples collected", len(samples)])

print(f"Saved to {OUTPUT_FILE}")
print(f"Average: {round(averageVal, 3)} ms")
print(f"Min: {round(minVal, 3)} ms")
print(f"Max: {round(maxVal, 3)} ms")




