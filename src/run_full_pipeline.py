import sys
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO


# ----------------------------------------------------
# Command line input
# Example: python src/run_full_pipeline.py SNMOT-060
# ----------------------------------------------------
if len(sys.argv) < 2:
    raise ValueError("Usage: python src/run_full_pipeline.py SNMOT-060")

CLIP_ID = sys.argv[1]

# ----------------------------------------------------
# Paths
# ----------------------------------------------------
FRAME_DIR = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/img1")

VIDEO_PATH = Path(f"data/raw_videos/{CLIP_ID}.mp4")
DETECTION_VIDEO_PATH = Path(f"outputs/videos/yolo_detections_{CLIP_ID}.mp4")
DETECTION_CSV_PATH = Path(f"outputs/detections/yolo_detections_{CLIP_ID}.csv")
RAW_TRACK_CSV_PATH = Path(f"outputs/tracks/raw_ball_track_{CLIP_ID}.csv")
IMPROVED_TRACK_CSV_PATH = Path(f"outputs/tracks/improved_ball_track_{CLIP_ID}.csv")
PLOT_PATH = Path(f"outputs/plots/ball_trajectory_{CLIP_ID}.png")
SUMMARY_PATH = Path(f"results/summary_{CLIP_ID}.csv")

for folder in [
    VIDEO_PATH.parent,
    DETECTION_VIDEO_PATH.parent,
    DETECTION_CSV_PATH.parent,
    RAW_TRACK_CSV_PATH.parent,
    IMPROVED_TRACK_CSV_PATH.parent,
    PLOT_PATH.parent,
    SUMMARY_PATH.parent,
]:
    folder.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------
# Settings
# ----------------------------------------------------
FPS = 25
MAX_JUMP_PIXELS = 80
MAX_FRAME_GAP = 8
MAX_INTERPOLATION_GAP = 5
SMOOTHING_WINDOW = 3


# ----------------------------------------------------
# Step 1: Make video from SoccerNet frames
# ----------------------------------------------------
print(f"\n[1/5] Creating video for {CLIP_ID}")

frames = sorted(FRAME_DIR.glob("*.jpg"))

if not frames:
    raise FileNotFoundError(f"No frames found in {FRAME_DIR}")

first_frame = cv2.imread(str(frames[0]))
height, width, _ = first_frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(VIDEO_PATH), fourcc, FPS, (width, height))

for i, frame_path in enumerate(frames):
    frame = cv2.imread(str(frame_path))
    out.write(frame)
    print(f"Writing frame {i + 1}/{len(frames)}", end="\r")

out.release()
print(f"\nSaved video: {VIDEO_PATH}")


# ----------------------------------------------------
# Step 2: Run YOLO and save detections
# ----------------------------------------------------
print(f"\n[2/5] Running YOLO on {CLIP_ID}")

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise Exception(f"Could not open video: {VIDEO_PATH}")

out = cv2.VideoWriter(str(DETECTION_VIDEO_PATH), fourcc, FPS, (width, height))

rows = []
frame_number = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model(frame, verbose=False)
    result = results[0]

    annotated_frame = result.plot()
    out.write(annotated_frame)

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            rows.append({
                "frame": frame_number,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            })

    print(f"Processed frame {frame_number}", end="\r")

cap.release()
out.release()

detections_df = pd.DataFrame(rows)
detections_df.to_csv(DETECTION_CSV_PATH, index=False)

print(f"\nSaved detections CSV: {DETECTION_CSV_PATH}")
print(f"Saved detection video: {DETECTION_VIDEO_PATH}")


# ----------------------------------------------------
# Step 3: Extract raw ball track
# ----------------------------------------------------
print(f"\n[3/5] Extracting raw ball track")

ball_df = detections_df[detections_df["class_name"] == "sports ball"].copy()

ball_track = (
    ball_df.sort_values("confidence", ascending=False)
    .groupby("frame", as_index=False)
    .first()
)

all_frames = pd.DataFrame({"frame": range(1, len(frames) + 1)})

ball_track = all_frames.merge(ball_track, on="frame", how="left")
ball_track["detected"] = ball_track["class_name"].notna()

ball_track.to_csv(RAW_TRACK_CSV_PATH, index=False)

print(f"Saved raw ball track: {RAW_TRACK_CSV_PATH}")


# ----------------------------------------------------
# Step 4: Improve ball track
# ----------------------------------------------------
print(f"\n[4/5] Improving ball track")

df = ball_track.copy()

df["raw_center_x"] = df["center_x"]
df["raw_center_y"] = df["center_y"]
df["raw_detected"] = df["detected"]

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

df["valid_detection"] = False
df.loc[valid_indices, "valid_detection"] = True

df.loc[~df["valid_detection"], ["center_x", "center_y"]] = np.nan

df["interp_center_x"] = df["center_x"].interpolate(
    limit=MAX_INTERPOLATION_GAP,
    limit_direction="both"
)

df["interp_center_y"] = df["center_y"].interpolate(
    limit=MAX_INTERPOLATION_GAP,
    limit_direction="both"
)

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

df.to_csv(IMPROVED_TRACK_CSV_PATH, index=False)

print(f"Saved improved ball track: {IMPROVED_TRACK_CSV_PATH}")


# ----------------------------------------------------
# Step 5: Create trajectory plot and summary
# ----------------------------------------------------
print(f"\n[5/5] Creating plot and summary")

raw_df = df[df["raw_detected"]]
improved_df = df[df["improved_available"]]

plt.figure(figsize=(12, 7))

plt.scatter(
    raw_df["raw_center_x"],
    raw_df["raw_center_y"],
    s=10,
    label="Raw YOLO Ball Detections"
)

plt.plot(
    improved_df["smooth_center_x"],
    improved_df["smooth_center_y"],
    linewidth=2,
    label="Motion-Constrained Smoothed Track"
)

plt.gca().invert_yaxis()
plt.title(f"Soccer Ball Trajectory - {CLIP_ID}")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()

plt.savefig(PLOT_PATH, bbox_inches="tight")
plt.close()

summary = pd.DataFrame([{
    "clip_id": CLIP_ID,
    "total_frames": len(df),
    "raw_detected_frames": int(df["raw_detected"].sum()),
    "valid_detected_frames": int(df["valid_detection"].sum()),
    "improved_available_frames": int(df["improved_available"].sum()),
    "raw_coverage_percent": df["raw_detected"].mean() * 100,
    "improved_coverage_percent": df["improved_available"].mean() * 100,
}])

summary.to_csv(SUMMARY_PATH, index=False)

print(f"Saved plot: {PLOT_PATH}")
print(f"Saved summary: {SUMMARY_PATH}")

print("\nPipeline complete.")
print(summary)