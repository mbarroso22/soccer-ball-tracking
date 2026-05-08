import sys
import pandas as pd
import numpy as np
from pathlib import Path

if len(sys.argv) < 2:
    raise ValueError("Usage: python src/11_evaluate_kalman.py SNMOT-060")

CLIP_ID = sys.argv[1]

BASE_DIR = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}")
GT_PATH = BASE_DIR / "gt" / "gt.txt"
GAMEINFO_PATH = BASE_DIR / "gameinfo.ini"
KALMAN_PATH = Path(f"outputs/tracks/kalman_ball_track_{CLIP_ID}.csv")
OUTPUT_PATH = Path(f"results/kalman_evaluation_{CLIP_ID}.csv")

ball_track_ids = set()

with open(GAMEINFO_PATH, "r", encoding="utf-8") as f:
    for line in f:
        lower = line.lower().strip()
        if "ball" in lower and lower.startswith("trackletid_"):
            left_side = lower.split("=")[0]
            track_id = int(left_side.replace("trackletid_", ""))
            ball_track_ids.add(track_id)

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
gt_ball["gt_center_x"] = gt_ball["x"] + gt_ball["w"] / 2
gt_ball["gt_center_y"] = gt_ball["y"] + gt_ball["h"] / 2

gt_ball = gt_ball.sort_values("frame").groupby("frame", as_index=False).first()

kalman = pd.read_csv(KALMAN_PATH)

merged = gt_ball.merge(kalman, on="frame", how="left")

merged["kalman_error_px"] = np.sqrt(
    (merged["kalman_center_x"] - merged["gt_center_x"]) ** 2
    + (merged["kalman_center_y"] - merged["gt_center_y"]) ** 2
)

available = merged[merged["kalman_available"] == True]

summary = pd.DataFrame([{
    "clip_id": CLIP_ID,
    "gt_ball_frames": len(gt_ball),
    "kalman_available_frames": len(available),
    "kalman_coverage_percent": len(available) / len(gt_ball) * 100 if len(gt_ball) > 0 else 0,
    "kalman_mean_error_px": available["kalman_error_px"].mean(),
    "kalman_median_error_px": available["kalman_error_px"].median(),
    "kalman_error_under_25px_percent": (available["kalman_error_px"] <= 25).mean() * 100,
    "kalman_error_under_50px_percent": (available["kalman_error_px"] <= 50).mean() * 100
}])

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
summary.to_csv(OUTPUT_PATH, index=False)

print(summary)
print(f"\nSaved: {OUTPUT_PATH}")