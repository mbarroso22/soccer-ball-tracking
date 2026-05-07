import pandas as pd
from pathlib import Path

INPUT_CSV = "outputs/detections/yolo_detections_SNMOT-060.csv"
OUTPUT_CSV = "outputs/tracks/raw_ball_track_SNMOT-060.csv"

Path("outputs/tracks").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

ball_df = df[df["class_name"] == "sports ball"].copy()

# Keep highest-confidence ball detection per frame
ball_track = (
    ball_df.sort_values("confidence", ascending=False)
    .groupby("frame", as_index=False)
    .first()
)

all_frames = pd.DataFrame({"frame": range(1, 751)})

ball_track = all_frames.merge(
    ball_track,
    on="frame",
    how="left"
)

ball_track["detected"] = ball_track["class_name"].notna()

ball_track.to_csv(OUTPUT_CSV, index=False)

print("Done.")
print(f"Saved raw ball track to: {OUTPUT_CSV}")
print(f"Total frames: {len(ball_track)}")
print(f"Detected frames: {ball_track['detected'].sum()}")
print(f"Missing frames: {(~ball_track['detected']).sum()}")
print(f"Detection coverage: {ball_track['detected'].mean() * 100:.2f}%")