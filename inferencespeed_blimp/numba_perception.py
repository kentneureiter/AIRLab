import numpy as np
import numba as nb
from numba import jit
import time

# Grid constants: keeping same as NiclaVision
N_ROWS = 14
N_COLS = 21
IMG_H = 160
IMG_W = 240
CELL_H = IMG_H // N_ROWS  
CELL_W = IMG_W // N_COLS

COLOR_DATA = {
    "purple": [[23.36, -39.03],  [[0.0648, 0.0560], [0.0560, 0.0590]]],
    "green":  [[-20.87, 8.57],   [[0.0602, 0.0489], [0.0489, 0.0647]]],
    "blue":   [[13.60, -33.79],  [[0.1300, 0.0665], [0.0665, 0.0402]]],
    "red":    [[53.57, 11.72],   [[0.0323, -0.0213], [-0.0213, 0.0373]]]
}

def rgb565_to_lab(raw_bytes):
    pixels = np.frombuffer(raw_bytes, dtype=np.uint16)
    R = ((pixels >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0)
    G = ((pixels >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0)
    B = ((pixels >> 0) & 0x1F).astype(np.float32) * (255.0 / 31.0)
    #by float32 did you mean like the [sign][exponent][mantissa] thing for each of these?

    R, G, B = R / 255.0, G / 255.0, B / 255.0
    def linearize(c):
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    R_lin = linearize(R)
    G_lin = linearize(G)
    B_lin = linearize(B)

    # RGB linear → XYZ (D65 illuminant)
    X = 0.4124 * R_lin + 0.3576 * G_lin + 0.1805 * B_lin
    Y = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
    Z = 0.0193 * R_lin + 0.1192 * G_lin + 0.9505 * B_lin

    # XYZ → LAB
    X, Y, Z = X / 0.95047, Y / 1.00000, Z / 1.08883  # normalize by D65 white point

    def f(t):
        return np.where(t > 0.008856, t ** (1/3), 7.787 * t + 16/116)

    L = 116 * f(Y) - 16
    A = 500 * (f(X) - f(Y))
    B_lab = 200 * (f(Y) - f(Z))
    return np.stack([L, A, B_lab], axis=1) #flips array from (38400 columns by 3 rows) to 38400 rows to 3 columns so that each row represents a pixel which is important for converting

def compute_cell_stats(lab):
    img   = lab.reshape(160, 240, 3)           # flat → image
    img   = img[:14*11, :21*11, :]       # crop to (154, 231, 3)
    grid  = img.reshape(14, 11, 21, 11, 3)     # image → cell grid
    grid  = grid.transpose(0, 2, 1, 3, 4)      # reorder axes → (14,21,11,11,3)
    cells = grid.reshape(294, 121, 3)           # collapse → 294 cells of 121 pixels
    means = np.mean(cells, axis=1)              # mean per cell → (294, 3)
    stds  = np.std(cells, axis=1)              # stdev per cell → (294, 3)
    return means, stds

@jit(nopython=True)
def compute_mahalanobis(means, mu, sigma_inv):
    diff = means - mu          # (294, 2) - how far each cell is from the color mean
    VQ   = diff @ sigma_inv    # (294, 2) - scale by inverse covariance  
    d2   = np.sum(VQ * diff, axis=1)   # (294,) - squared Mahalanobis distance per cell
    return np.exp(-d2 / 2.0)  


def run_perception(raw_bytes):
    # Step 1: convert frame to LAB
    lab = rgb565_to_lab(raw_bytes)
    
    # Step 2: compute cell stats
    means, stds = compute_cell_stats(lab)
    ab_means = means[:, 1:3]  # just A and B columns → (294, 2)
    
    # Step 3: score each color
    scores = {}
    for color, (mu, sigma) in COLOR_DATA.items():
        mu = np.array(mu, dtype=np.float32)
        sigma_inv = np.linalg.inv(np.array(sigma, dtype=np.float32))
        scores[color] = compute_mahalanobis(ab_means, mu, sigma_inv)
    
    return scores

# --- Benchmark ---
# Simulate a fake frame (random RGB565 bytes)
fake_frame = np.random.randint(0, 65535, IMG_H * IMG_W, dtype=np.uint16).tobytes()

# Warm up Numba (first call triggers compilation)
print("Warming up Numba...")
_ = run_perception(fake_frame)

# Now benchmark
N = 100
start = time.time()
for _ in range(N):
    run_perception(fake_frame)
elapsed = (time.time() - start) / N * 1000

print(f"Average inference time (Mac+Numba): {elapsed:.2f} ms")
print(f"NiclaVision baseline: ~104 ms")
print(f"Speedup: {104 / elapsed:.1f}x")


