import cv2
from pathlib import Path

FRAME_DIR = Path("SoccerTrackingData/SoccerNet/tracking/train/train/SNMOT-060/img1")
OUTPUT_PATH = Path("data/raw_videos/SNMOT-060.mp4")

FPS = 25

frames = sorted(FRAME_DIR.glob("*.jpg"))

if not frames:
    raise FileNotFoundError(f"No frames found in {FRAME_DIR}")

first_frame = cv2.imread(str(frames[0]))
height, width, _ = first_frame.shape

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (width, height))

for i, frame_path in enumerate(frames):
    frame = cv2.imread(str(frame_path))
    out.write(frame)

    print(f"Writing frame {i + 1}/{len(frames)}", end="\r")

out.release()

print(f"\nSaved video to {OUTPUT_PATH}")