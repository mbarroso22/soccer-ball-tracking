import sys
import cv2
from pathlib import Path

print("SCRIPT STARTED")

if len(sys.argv) < 5:
    raise ValueError(
        "Usage: python src/16_extract_broadcast_clip.py "
        '"INPUT_MKV_PATH" OUTPUT_NAME START_SECONDS DURATION_SECONDS'
    )

INPUT_PATH = Path(sys.argv[1])
OUTPUT_NAME = sys.argv[2]
START_SECONDS = float(sys.argv[3])
DURATION_SECONDS = float(sys.argv[4])

OUTPUT_PATH = Path(f"data/raw_videos/{OUTPUT_NAME}.mp4")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

print(f"Input path: {INPUT_PATH}")
print(f"Input exists: {INPUT_PATH.exists()}")
print(f"Output path: {OUTPUT_PATH}")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Input file does not exist: {INPUT_PATH}")

cap = cv2.VideoCapture(str(INPUT_PATH))

print(f"Video opened: {cap.isOpened()}")

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {INPUT_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Resolution: {width}x{height}")
print(f"Total frames: {total_frames}")

start_frame = int(START_SECONDS * fps)
end_frame = int((START_SECONDS + DURATION_SECONDS) * fps)

print(f"Start frame: {start_frame}")
print(f"End frame: {end_frame}")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

print(f"Video writer opened: {out.isOpened()}")

if not out.isOpened():
    raise RuntimeError(f"Could not open output writer: {OUTPUT_PATH}")

current_frame = start_frame
written = 0

while current_frame < end_frame:
    ret, frame = cap.read()

    if not ret:
        print("No more frames available.")
        break

    out.write(frame)

    current_frame += 1
    written += 1

    if written % 25 == 0:
        print(f"Written {written} frames...")

cap.release()
out.release()

print(f"Saved broadcast clip: {OUTPUT_PATH}")
print(f"Total written frames: {written}")