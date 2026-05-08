import sys
import cv2
import pandas as pd
from pathlib import Path

if len(sys.argv) < 2:
    raise ValueError("Usage: python src/09_create_demo_video.py SNMOT-060")

CLIP_ID = sys.argv[1]

FRAME_DIR = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/img1")
GT_PATH = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/gt/gt.txt")
GAMEINFO_PATH = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}/gameinfo.ini")
YOLO_PATH = Path(f"outputs/detections/yolo_detections_{CLIP_ID}.csv")
IMPROVED_PATH = Path(f"outputs/tracks/improved_ball_track_{CLIP_ID}.csv")
OUTPUT_PATH = Path(f"outputs/videos/final_demo_{CLIP_ID}.mp4")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FPS = 25

# -----------------------------
# Read object labels from gameinfo
# -----------------------------
track_labels = {}

with open(GAMEINFO_PATH, "r", encoding="utf-8") as f:
    for line in f:
        lower = line.lower().strip()

        if lower.startswith("trackletid_"):
            left, right = line.strip().split("=", 1)
            track_id = int(left.lower().replace("trackletid_", ""))
            track_labels[track_id] = right.strip()

ball_track_ids = {
    tid for tid, label in track_labels.items()
    if "ball" in label.lower()
}

# -----------------------------
# Load ground truth tracks
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

gt["x1"] = gt["x"]
gt["y1"] = gt["y"]
gt["x2"] = gt["x"] + gt["w"]
gt["y2"] = gt["y"] + gt["h"]

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
# Output setup
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

    frame_gt = gt[gt["frame"] == frame_num]

    # -----------------------------
    # Draw player/referee/goalkeeper boxes from GT
    # -----------------------------
    for _, row in frame_gt.iterrows():
        track_id = int(row["track_id"])
        label = track_labels.get(track_id, "unknown")

        if track_id in ball_track_ids:
            continue

        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

        if "referee" in label.lower():
            color = (0, 255, 255)  # yellow
            short_label = f"Ref {track_id}"
        elif "goalkeeper" in label.lower() or "goalkeepers" in label.lower():
            color = (255, 255, 0)  # cyan
            short_label = f"GK {track_id}"
        else:
            color = (255, 255, 255)  # white
            short_label = f"Player {track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame,
            short_label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        )

    # -----------------------------
    # Draw GT ball box
    # -----------------------------
    ball_rows = frame_gt[frame_gt["track_id"].isin(ball_track_ids)]

    for _, row in ball_rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            frame,
            "GT Ball",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    # -----------------------------
    # Draw YOLO ball prediction
    # -----------------------------
    yolo_rows = yolo_ball[yolo_ball["frame"] == frame_num]

    for _, row in yolo_rows.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        conf = row["confidence"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            f"YOLO Ball {conf:.2f}",
            (x1, y2 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    # -----------------------------
    # Draw improved ball trajectory
    # -----------------------------
    imp_rows = improved[improved["frame"] == frame_num]

    if not imp_rows.empty:
        row = imp_rows.iloc[0]

        if bool(row["improved_available"]):
            cx = int(row["smooth_center_x"])
            cy = int(row["smooth_center_y"])

            trail_points.append((cx, cy))
            trail_points = trail_points[-30:]

            cv2.circle(frame, (cx, cy), 7, (255, 0, 0), -1)

    for i in range(1, len(trail_points)):
        cv2.line(frame, trail_points[i - 1], trail_points[i], (255, 0, 0), 2)

    # -----------------------------
    # Header / legend
    # -----------------------------
    cv2.rectangle(frame, (0, 0), (width, 75), (0, 0, 0), -1)

    cv2.putText(
        frame,
        f"{CLIP_ID} | Frame {frame_num} | Soccer Ball Tracking Demo",
        (25, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "White: Players | Yellow: Referees | Green: GT Ball | Red: YOLO Ball | Blue: Smoothed Track",
        (25, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    out.write(frame)
    print(f"Demo frame {frame_num}/{len(frames)}", end="\r")

out.release()

print(f"\nSaved demo video: {OUTPUT_PATH}")