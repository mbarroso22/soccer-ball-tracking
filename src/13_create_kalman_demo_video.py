import sys
import cv2
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    raise ValueError("Usage: python src/13_create_kalman_demo_video.py SNMOT-060")

CLIP_ID = sys.argv[1]

FRAME_DIR = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/img1")
GT_PATH = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/gt/gt.txt")
GAMEINFO_PATH = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/gameinfo.ini")
YOLO_PATH = Path(f"outputs/detections/yolo_detections_{CLIP_ID}.csv")
IMPROVED_PATH = Path(f"outputs/tracks/improved_ball_track_{CLIP_ID}.csv")
KALMAN_PATH = Path(f"outputs/tracks/kalman_ball_track_{CLIP_ID}.csv")
OUTPUT_PATH = Path(f"outputs/videos/kalman_demo_{CLIP_ID}.mp4")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FPS = 25

# -----------------------------
# Get ball track IDs from gameinfo.ini
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
# Load GT ball boxes
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
gt_ball["gt_center_x"] = gt_ball["x"] + gt_ball["w"] / 2
gt_ball["gt_center_y"] = gt_ball["y"] + gt_ball["h"] / 2

# -----------------------------
# Load YOLO detections
# -----------------------------
yolo = pd.read_csv(YOLO_PATH)
yolo_ball = yolo[yolo["class_name"] == "sports ball"].copy()
yolo_ball = (
    yolo_ball.sort_values("confidence", ascending=False)
    .groupby("frame", as_index=False)
    .first()
)

# -----------------------------
# Load smoothed and Kalman tracks
# -----------------------------
improved = pd.read_csv(IMPROVED_PATH)
kalman = pd.read_csv(KALMAN_PATH)

# -----------------------------
# Prepare video
# -----------------------------
frames = sorted(FRAME_DIR.glob("*.jpg"))

if not frames:
    raise FileNotFoundError(f"No frames found in {FRAME_DIR}")

first = cv2.imread(str(frames[0]))
height, width, _ = first.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (width, height))

smooth_trail = []
kalman_trail = []
gt_trail = []

for frame_path in frames:
    frame_num = int(frame_path.stem)
    frame = cv2.imread(str(frame_path))

    # -----------------------------
    # Ground truth ball
    # Green
    # -----------------------------
    gt_rows = gt_ball[gt_ball["frame"] == frame_num]

    for _, row in gt_rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        gcx, gcy = int(row["gt_center_x"]), int(row["gt_center_y"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(frame, (gcx, gcy), 5, (0, 255, 0), -1)
        gt_trail.append((gcx, gcy))

        cv2.putText(
            frame,
            "GT",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # Raw YOLO ball detection
    # Red
    # -----------------------------
    yolo_rows = yolo_ball[yolo_ball["frame"] == frame_num]

    for _, row in yolo_rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        conf = row["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"YOLO {conf:.2f}",
            (x1, y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    # -----------------------------
    # Smoothed/interpolated track
    # Blue
    # -----------------------------
    imp_rows = improved[improved["frame"] == frame_num]

    if not imp_rows.empty:
        row = imp_rows.iloc[0]

        if bool(row["improved_available"]):
            sx = int(row["smooth_center_x"])
            sy = int(row["smooth_center_y"])

            smooth_trail.append((sx, sy))
            cv2.circle(frame, (sx, sy), 6, (255, 0, 0), -1)

    # -----------------------------
    # Kalman track
    # Purple
    # -----------------------------
    kal_rows = kalman[kalman["frame"] == frame_num]

    if not kal_rows.empty:
        row = kal_rows.iloc[0]

        if bool(row["kalman_available"]):
            kx = int(row["kalman_center_x"])
            ky = int(row["kalman_center_y"])

            kalman_trail.append((kx, ky))
            cv2.circle(frame, (kx, ky), 6, (255, 0, 255), -1)

    # Keep trails readable
    gt_trail = gt_trail[-20:]
    smooth_trail = smooth_trail[-30:]
    kalman_trail = kalman_trail[-30:]

    for i in range(1, len(gt_trail)):
        cv2.line(frame, gt_trail[i - 1], gt_trail[i], (0, 255, 0), 2)

    for i in range(1, len(smooth_trail)):
        cv2.line(frame, smooth_trail[i - 1], smooth_trail[i], (255, 0, 0), 2)

    for i in range(1, len(kalman_trail)):
        cv2.line(frame, kalman_trail[i - 1], kalman_trail[i], (255, 0, 255), 2)

    # -----------------------------
    # Header
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (width, 95), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"{CLIP_ID} | Frame {frame_num} | Ball Tracking Comparison",
        (25, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Green: Ground Truth | Red: YOLO Detection | Blue: Motion-Smoothed | Purple: Kalman Prediction",
        (25, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Goal: compare raw detector output with temporal post-processing under missed ball detections",
        (25, 86),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (210, 210, 210),
        1
    )

    out.write(frame)
    print(f"Kalman demo frame {frame_num}/{len(frames)}", end="\r")

out.release()

print(f"\nSaved Kalman demo video: {OUTPUT_PATH}")