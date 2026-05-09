import sys
import pandas as pd
import numpy as np
from pathlib import Path
from filterpy.kalman import KalmanFilter

if len(sys.argv) < 2:
    raise ValueError("Usage: python src/10_kalman_ball_track.py SNMOT-060")

CLIP_ID = sys.argv[1]

INPUT_PATH = Path(f"outputs/tracks/improved_ball_track_{CLIP_ID}.csv")
OUTPUT_PATH = Path(f"outputs/tracks/kalman_ball_track_{CLIP_ID}.csv")

df = pd.read_csv(INPUT_PATH)

# -----------------------------
# Tunable parameters
# -----------------------------
MIN_CONFIDENCE = 0.25
MAX_MISSING_FRAMES = 20

PROCESS_NOISE = 5.0
MEASUREMENT_NOISE = 15.0
INITIAL_UNCERTAINTY = 500.0

# State: x, y, vx, vy
kf = KalmanFilter(dim_x=4, dim_z=2)

kf.F = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=float)

kf.H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
], dtype=float)

kf.P *= INITIAL_UNCERTAINTY
kf.R *= MEASUREMENT_NOISE
kf.Q *= PROCESS_NOISE

initialized = False
missing_count = 0

kalman_x = []
kalman_y = []
kalman_available = []
kalman_mode = []

for _, row in df.iterrows():
    has_measurement = bool(row["improved_available"])

    # Confidence gate: reject weak YOLO ball detections before Kalman update
    if has_measurement and "confidence" in df.columns:
        if pd.notna(row["confidence"]) and row["confidence"] < MIN_CONFIDENCE:
            has_measurement = False

    if has_measurement:
        z = np.array([row["smooth_center_x"], row["smooth_center_y"]])

        if not initialized:
            kf.x = np.array([z[0], z[1], 0, 0], dtype=float)
            initialized = True
        else:
            kf.predict()
            kf.update(z)

        missing_count = 0

        kalman_x.append(float(kf.x[0]))
        kalman_y.append(float(kf.x[1]))
        kalman_available.append(True)
        kalman_mode.append("measurement_update")

    else:
        if initialized and missing_count < MAX_MISSING_FRAMES:
            kf.predict()
            missing_count += 1

            kalman_x.append(float(kf.x[0]))
            kalman_y.append(float(kf.x[1]))
            kalman_available.append(True)
            kalman_mode.append("prediction_only")
        else:
            kalman_x.append(np.nan)
            kalman_y.append(np.nan)
            kalman_available.append(False)
            kalman_mode.append("unavailable")

df["kalman_center_x"] = kalman_x
df["kalman_center_y"] = kalman_y
df["kalman_available"] = kalman_available
df["kalman_mode"] = kalman_mode

df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved Kalman track: {OUTPUT_PATH}")
print(f"Kalman available frames: {df['kalman_available'].sum()}")
print(f"Kalman coverage: {df['kalman_available'].mean() * 100:.2f}%")
print("\nKalman mode counts:")
print(df["kalman_mode"].value_counts())