import serial
import numpy as np
import time
import csv
from numba_perception import run_perception
import cv2

# ---- CONFIG ----
# Run this in terminal first to find your port:
#   ls /dev/cu.usbmodem*
PORT = '/dev/cu.usbmodem395B355A30312'  # update this if different
N_FRAMES = 200
FRAME_SIZE = 240 * 160 * 2     # 76800 bytes
MAGIC = b'\xFF\xFE\xFD\xFC'
# ----------------

# Warm up Numba before we start receiving (avoids counting compile time)
print("Warming up Numba...")
fake = np.random.randint(0, 65535, 240 * 160, dtype=np.uint16).tobytes()
run_perception(fake)
print("Numba ready.")

# Open serial port
ser = serial.Serial(PORT, 115200, timeout=10)
print(f"Listening on {PORT} — start NiclaVision now\n")

results = []

for i in range(N_FRAMES):
    # Wait for magic header byte by byte
    buf = b''
    while buf != MAGIC:
        byte = ser.read(1)
        buf = (buf + byte)[-4:]

    # Time how long the frame transfer takes
    t_transfer_start = time.time()
    raw = ser.read(FRAME_SIZE)
    transfer_ms = (time.time() - t_transfer_start) * 1000

    # Time Mac-side inference
    t_inf_start = time.time()
    scores = run_perception(raw)
    inference_ms = (time.time() - t_inf_start) * 1000

    # Acknowledge so NiclaVision sends next frame
    ser.write(b'\x01')

    results.append((i, transfer_ms, inference_ms))
    print(f"Frame {i+1:>3}/{N_FRAMES} | Transfer: {transfer_ms:6.1f}ms | Inference: {inference_ms:.2f}ms | Total: {transfer_ms+inference_ms:.1f}ms")

# Save CSV
with open('inference_log_mac.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['frame', 'transfer_ms', 'inference_ms', 'total_ms'])
    for frame, t, inf in results:
        writer.writerow([frame, round(t, 2), round(inf, 2), round(t + inf, 2)])

avg_t   = sum(r[1] for r in results) / N_FRAMES
avg_inf = sum(r[2] for r in results) / N_FRAMES
print(f"\nSaved to inference_log_mac.csv")
print(f"Avg transfer:  {avg_t:.1f}ms")
print(f"Avg inference: {avg_inf:.2f}ms")
print(f"Avg total:     {avg_t + avg_inf:.1f}ms")
print(f"NiclaVision baseline: 104ms")