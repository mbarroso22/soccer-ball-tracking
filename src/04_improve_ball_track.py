import pandas as pd
import numpy as np
from pathlib import Path

INPUT_CSV = "outputs/tracks/raw_ball_track_SNMOT-060.csv"
OUTPUT_CSV = "outputs/tracks/improved_ball_track_v2_SNMOT-060.csv"

MAX_JUMP_PIXELS = 80
MAX_FRAME_GAP = 8
MAX_INTERPOLATION_GAP = 5
SMOOTHING_WINDOW = 3

Path("outputs/tracks").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

df["raw_center_x"] = df["center_x"]
df["raw_center_y"] = df["center_y"]
df["raw_detected"] = df["detected"]

# --------------------------------------------------
# Step 1: keep only valid detections
# --------------------------------------------------
valid_indices = []

last_x = None
last_y = None
last_frame = None

for i, row in df.iterrows():

    if not row["raw_detected"]:
        continue

    x = row["raw_center_x"]
    y = row["raw_center_y"]
    frame = row["frame"]

    keep = True

    if last_x is not None:

        dist = np.sqrt((x - last_x) ** 2 + (y - last_y) ** 2)
        frame_gap = frame - last_frame

        if dist > MAX_JUMP_PIXELS or frame_gap > MAX_FRAME_GAP:
            keep = False

    if keep:
        valid_indices.append(i)
        last_x = x
        last_y = y
        last_frame = frame

# --------------------------------------------------
# Step 2: create clean track
# --------------------------------------------------
df["valid_detection"] = False
df.loc[valid_indices, "valid_detection"] = True

df.loc[~df["valid_detection"], ["center_x", "center_y"]] = np.nan

# --------------------------------------------------
# Step 3: interpolate small gaps only
# --------------------------------------------------
df["interp_center_x"] = df["center_x"].interpolate(
    limit=MAX_INTERPOLATION_GAP,
    limit_direction="both"
)

df["interp_center_y"] = df["center_y"].interpolate(
    limit=MAX_INTERPOLATION_GAP,
    limit_direction="both"
)

# --------------------------------------------------
# Step 4: smoothing
# --------------------------------------------------
df["smooth_center_x"] = df["interp_center_x"].rolling(
    window=SMOOTHING_WINDOW,
    center=True,
    min_periods=1
).mean()

df["smooth_center_y"] = df["interp_center_y"].rolling(
    window=SMOOTHING_WINDOW,
    center=True,
    min_periods=1
).mean()

df["improved_available"] = (
    df["smooth_center_x"].notna()
    & df["smooth_center_y"].notna()
)

df.to_csv(OUTPUT_CSV, index=False)

print("Done.")
print(f"Saved improved track: {OUTPUT_CSV}")
print(f"Raw detections: {df['raw_detected'].sum()}")
print(f"Valid detections: {df['valid_detection'].sum()}")
print(f"Improved frames: {df['improved_available'].sum()}")
print(f"Coverage: {df['improved_available'].mean() * 100:.2f}%")