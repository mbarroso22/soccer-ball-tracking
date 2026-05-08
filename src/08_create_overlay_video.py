import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    raise ValueError("Usage: python src/08_create_overlay_video.py SNMOT-060")

CLIP_ID = sys.argv[1]

FRAME_DIR = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/img1")
GT_PATH = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/gt/gt.txt")
GAMEINFO_PATH = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/gameinfo.ini")
YOLO_PATH = Path(f"outputs/detections/yolo_detections_{CLIP_ID}.csv")
IMPROVED_PATH = Path(f"outputs/tracks/improved_ball_track_{CLIP_ID}.csv")
OUTPUT_PATH = Path(f"outputs/videos/overlay_comparison_{CLIP_ID}.mp4")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FPS = 25

# -----------------------------
# Find ball track IDs
# -----------------------------
ball_track_ids = set()

with open(GAMEINFO_PATH, "r", encoding="utf-8") as f:
    for line in f:
        lower = line.lower().strip()
        if "ball" in lower and lower.startswith("trackletid_"):
            left_side = lower.split("=")[0]
            track_id = int(left_side.replace("trackletid_", ""))
            ball_track_ids.add(track_id)

print(f"Ball track IDs: {sorted(ball_track_ids)}")

# -----------------------------
# Load GT
# -----------------------------
gt = pd.read_csv(GT_PATH, header=None)
gt = gt.rename(columns={
    0: "frame",
    1: "track_id",
    2: "x",
    3: "y",
    4: "w",
    5: "h",
    6: "conf"
})

gt_ball = gt[gt["track_id"].isin(ball_track_ids)].copy()
gt_ball["x1"] = gt_ball["x"]
gt_ball["y1"] = gt_ball["y"]
gt_ball["x2"] = gt_ball["x"] + gt_ball["w"]
gt_ball["y2"] = gt_ball["y"] + gt_ball["h"]

# -----------------------------
# Load YOLO ball detections
# -----------------------------
yolo = pd.read_csv(YOLO_PATH)
yolo_ball = yolo[yolo["class_name"] == "sports ball"].copy()

yolo_ball = (
    yolo_ball.sort_values("confidence", ascending=False)
    .groupby("frame", as_index=False)
    .first()
)

# -----------------------------
# Load improved trajectory
# -----------------------------
improved = pd.read_csv(IMPROVED_PATH)

# -----------------------------
# Prepare output video
# -----------------------------
frames = sorted(FRAME_DIR.glob("*.jpg"))

if not frames:
    raise FileNotFoundError(f"No frames found in {FRAME_DIR}")

first = cv2.imread(str(frames[0]))
height, width, _ = first.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (width, height))

trail_points = []

for frame_path in frames:
    frame_num = int(frame_path.stem)
    frame = cv2.imread(str(frame_path))

    # -----------------------------
    # Draw ground-truth ball box
    # Green
    # -----------------------------
    gt_rows = gt_ball[gt_ball["frame"] == frame_num]
    for _, row in gt_rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "GT Ball", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # -----------------------------
    # Draw raw YOLO ball detection
    # Red
    # -----------------------------
    yolo_rows = yolo_ball[yolo_ball["frame"] == frame_num]
    for _, row in yolo_rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        conf = row["confidence"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"YOLO Ball {conf:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # -----------------------------
    # Draw improved smoothed center
    # Blue circle and trail
    # -----------------------------
    imp_rows = improved[improved["frame"] == frame_num]

    if not imp_rows.empty:
        row = imp_rows.iloc[0]

        if bool(row["improved_available"]):
            cx = int(row["smooth_center_x"])
            cy = int(row["smooth_center_y"])

            trail_points.append((cx, cy))

            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
            cv2.putText(frame, "Improved Track", (cx + 8, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Keep trail short
    trail_points = trail_points[-25:]

    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i - 1], trail_points[i], (255, 0, 0), 2)

    # -----------------------------
    # Legend
    # -----------------------------
    cv2.putText(frame, "Green: Ground Truth | Red: YOLO | Blue: Improved Track",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(frame, "Green: Ground Truth | Red: YOLO | Blue: Improved Track",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)

    out.write(frame)
    print(f"Overlay frame {frame_num}/{len(frames)}", end="\r")

out.release()

print(f"\nSaved overlay video: {OUTPUT_PATH}")