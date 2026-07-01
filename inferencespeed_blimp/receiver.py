import serial
import numpy as np
import time
import cv2
from numba_perception import run_perception

# ---- CONFIG ----
PORT       = '/dev/cu.usbmodem395B355A30312'  # update if needed
FRAME_SIZE = 240 * 160 * 2
MAGIC      = b'\xFF\xFE\xFD\xFC'
THRESHOLD  = 0.3    # min score to draw a cell
SCALE      = 4      # display scale (240x160 -> 960x640)
# ----------------

# Color overlays in BGR (OpenCV uses BGR not RGB)
COLOR_BGR = {
    "purple": (128, 0, 128),
    "green":  (0, 200, 0),
    "blue":   (255, 80, 0),
    "red":    (0, 0, 220),
}

N_ROWS, N_COLS = 14, 21
CELL_H = 160 // N_ROWS   # 11px
CELL_W = 240 // N_COLS   # 11px


def decode_frame(raw):
    """Convert raw RGB565 bytes to a BGR uint8 image for OpenCV."""
    pixels = np.frombuffer(raw, dtype=np.uint16)
    r = (((pixels >> 11) & 0x1F) * (255 / 31)).astype(np.uint8)
    g = (((pixels >> 5)  & 0x3F) * (255 / 63)).astype(np.uint8)
    b = (( pixels        & 0x1F) * (255 / 31)).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=1).reshape(160, 240, 3)
    return rgb[:, :, ::-1].copy()  # RGB → BGR, make contiguous


def draw_scores(frame, scores):
    """Draw colored rectangles on cells above the score threshold."""
    for color, score_array in scores.items():
        bgr = COLOR_BGR[color]
        best_score = float(score_array.max())
        best_idx   = int(score_array.argmax())

        for idx in range(N_ROWS * N_COLS):
            score = float(score_array[idx])
            if score < THRESHOLD:
                continue
            row = idx // N_COLS
            col = idx % N_COLS
            x1, y1 = col * CELL_W, row * CELL_H
            x2, y2 = x1 + CELL_W, y1 + CELL_H
            alpha = int(score * 255)  # brighter = more confident
            overlay_color = tuple(int(c * score) for c in bgr)
            cv2.rectangle(frame, (x1, y1), (x2, y2), overlay_color, thickness=1)

        # Draw centroid circle on the best cell if above threshold
        if best_score >= THRESHOLD:
            row = best_idx // N_COLS
            col = best_idx % N_COLS
            cx  = col * CELL_W + CELL_W // 2
            cy  = row * CELL_H + CELL_H // 2
            cv2.circle(frame, (cx, cy), 6, bgr, thickness=2)
            cv2.putText(frame, color, (cx - 20, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, bgr, 1)
    return frame


# Warm up Numba
print("loading in Numba")
fake = np.random.randint(0, 65535, 240 * 160, dtype=np.uint16).tobytes()
run_perception(fake)
print("Numba ready.")

ser = serial.Serial(PORT, 115200, timeout=10)
print(f"Listening on {PORT} — start NiclaVision now\n")

frame_count = 0

while True:
    # Wait for magic header
    buf = b''
    while buf != MAGIC:
        byte = ser.read(1)
        if not byte:
            continue
        buf = (buf + byte)[-4:]

    # Receive frame
    raw = ser.read(FRAME_SIZE)
    ser.write(b'\x01')  # ack

    # Decode to image
    bgr = decode_frame(raw)

    # Run inference
    t0 = time.time()
    scores = run_perception(raw)
    inf_ms = (time.time() - t0) * 1000

    # Draw cell scores on image
    draw_scores(bgr, scores)

    # Add FPS / inference time overlay
    frame_count += 1
    cv2.putText(bgr, f"Inference: {inf_ms:.1f}ms", (4, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.putText(bgr, f"Frame: {frame_count}", (4, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Scale up and show (240x160 is tiny)
    display = cv2.resize(bgr, (240 * SCALE, 160 * SCALE),
                         interpolation=cv2.INTER_NEAREST)
    cv2.imshow('NiclaVision Live Inference', display)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
cv2.destroyAllWindows()
print("Done.")