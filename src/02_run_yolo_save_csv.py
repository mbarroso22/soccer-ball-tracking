import cv2
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

VIDEO_PATH = "data/raw_videos/SNMOT-060.mp4"
OUTPUT_VIDEO_PATH = "outputs/videos/yolo_detections_SNMOT-060.mp4"
OUTPUT_CSV_PATH = "outputs/detections/yolo_detections_SNMOT-060.csv"

model = YOLO("yolov8n.pt")

Path("outputs/videos").mkdir(parents=True, exist_ok=True)
Path("outputs/detections").mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception(f"Could not open video: {VIDEO_PATH}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

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

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            rows.append({
                "frame": frame_number,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": center_x,
                "center_y": center_y
            })

    print(f"Processed frame {frame_number}", end="\r")

cap.release()
out.release()

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV_PATH, index=False)

print("\nDone.")
print(f"Saved video: {OUTPUT_VIDEO_PATH}")
print(f"Saved CSV: {OUTPUT_CSV_PATH}")
print(f"Total detections: {len(df)}")
print(df["class_name"].value_counts())