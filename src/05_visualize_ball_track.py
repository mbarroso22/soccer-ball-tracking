import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

INPUT_CSV = "outputs/tracks/improved_ball_track_v2_SNMOT-060.csv"

Path("outputs/plots").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_CSV)

# ----------------------------------------
# Raw detections
# ----------------------------------------
raw_df = df[df["raw_detected"]]

# ----------------------------------------
# Improved trajectory
# ----------------------------------------
improved_df = df[df["improved_available"]]

# ----------------------------------------
# Plot
# ----------------------------------------
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
    label="Improved Smoothed Trajectory"
)

plt.gca().invert_yaxis()

plt.title("Soccer Ball Trajectory")
plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.legend()

OUTPUT_PATH = "outputs/plots/ball_trajectory_plot.png"

plt.savefig(OUTPUT_PATH, bbox_inches="tight")

print(f"Saved plot to: {OUTPUT_PATH}")