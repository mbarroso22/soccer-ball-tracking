from ultralytics import YOLO
import cv2
import pandas as pd
from pathlib import Path

VIDEO_PATH = "data/raw_videos/liverpool_manutd_clip.mp4"

OUTPUT_VIDEO = "outputs/videos/broadcast_yolo_demo.mp4"
OUTPUT_CSV = "outputs/detections/broadcast_yolo_detections.csv"

Path("outputs/videos").mkdir(parents=True, exist_ok=True)
Path("outputs/detections").mkdir(parents=True, exist_ok=True)

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)

all_detections = []

frame_num = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        all_detections.append({
            "frame": frame_num,
            "class_name": class_name,
            "confidence": confidence,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2
        })

        # Highlight sports ball strongly
        if class_name == "sports ball":
            color = (0, 0, 255)
            thickness = 3
        else:
            color = (0, 255, 0)
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        cv2.putText(
            frame,
            f"{class_name} {confidence:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    cv2.putText(
        frame,
        "Broadcast Soccer Ball Tracking Demo",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    out.write(frame)

    frame_num += 1

    if frame_num % 25 == 0:
        print(f"Processed {frame_num}/{total_frames} frames")

cap.release()
out.release()

df = pd.DataFrame(all_detections)
df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved video: {OUTPUT_VIDEO}")
print(f"Saved detections: {OUTPUT_CSV}")

if len(df) > 0:
    print("\nDetection counts:")
    print(df["class_name"].value_counts())