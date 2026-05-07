import cv2
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
VIDEO_PATH = "data/raw_videos/SNMOT-060.mp4"
OUTPUT_PATH = "outputs/videos/baseline_SNMOT-060.mp4"

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolov8n.pt")

# -----------------------------
# Open video
# -----------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception(f"Could not open video: {VIDEO_PATH}")

# -----------------------------
# Video properties
# -----------------------------
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# -----------------------------
# Video writer
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

# -----------------------------
# Process frames
# -----------------------------
frame_count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Save frame
    out.write(annotated_frame)

    frame_count += 1

    print(f"Processed frame: {frame_count}", end="\r")

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
out.release()

print("\nDone.")
print(f"Saved output to: {OUTPUT_PATH}")