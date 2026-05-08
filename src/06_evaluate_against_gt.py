import sys
import configparser
import pandas as pd
import numpy as np
from pathlib import Path


if len(sys.argv) < 2:
    raise ValueError("Usage: python src/06_evaluate_against_gt.py SNMOT-060")

CLIP_ID = sys.argv[1]

BASE_DIR = Path(f"SoccerTrackingData/SoccerNet/tracking/train/train/{CLIP_ID}")
GT_PATH = BASE_DIR / "gt" / "gt.txt"
GAMEINFO_PATH = BASE_DIR / "gameinfo.ini"

YOLO_PATH = Path(f"outputs/detections/yolo_detections_{CLIP_ID}.csv")
IMPROVED_PATH = Path(f"outputs/tracks/improved_ball_track_{CLIP_ID}.csv")
OUTPUT_PATH = Path(f"results/evaluation_{CLIP_ID}.csv")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union == 0:
        return 0

    return intersection / union


# ----------------------------------------------------
# Load ground truth
# MOT format:
# frame, track_id, x, y, w, h, conf, ...
# ----------------------------------------------------
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


# ----------------------------------------------------
# Identify ball track IDs from gameinfo.ini
# This is intentionally flexible because SoccerNet metadata
# can vary slightly by sequence.
# ----------------------------------------------------
ball_track_ids = set()

with open(GAMEINFO_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    lower = line.lower().strip()

    if "ball" in lower and lower.startswith("trackletid_"):
        left_side = lower.split("=")[0]
        track_id = int(left_side.replace("trackletid_", ""))
        ball_track_ids.add(track_id)

if not ball_track_ids:
    print("Could not automatically identify ball track ID from gameinfo.ini.")
    print("Open gameinfo.ini and look for the ball track ID.")
    raise SystemExit

print(f"Ball track IDs found: {sorted(ball_track_ids)}")

gt_ball = gt[gt["track_id"].isin(ball_track_ids)].copy()

if gt_ball.empty:
    raise ValueError("No ground-truth ball boxes found. Ball ID parsing may be wrong.")

# Keep one GT ball box per frame
gt_ball = gt_ball.sort_values("frame").groupby("frame", as_index=False).first()


# ----------------------------------------------------
# Load YOLO sports ball detections
# ----------------------------------------------------
yolo = pd.read_csv(YOLO_PATH)
yolo_ball = yolo[yolo["class_name"] == "sports ball"].copy()

# Keep highest-confidence YOLO ball per frame
yolo_ball = (
    yolo_ball.sort_values("confidence", ascending=False)
    .groupby("frame", as_index=False)
    .first()
)


# ----------------------------------------------------
# Evaluate raw YOLO detections against GT
# ----------------------------------------------------
IOU_THRESHOLD = 0.5

raw_matches = 0
raw_total_iou = []
raw_pred_frames = set(yolo_ball["frame"].tolist())

for _, gt_row in gt_ball.iterrows():
    frame = gt_row["frame"]
    pred = yolo_ball[yolo_ball["frame"] == frame]

    if pred.empty:
        continue

    pred_row = pred.iloc[0]

    gt_box = [gt_row["x1"], gt_row["y1"], gt_row["x2"], gt_row["y2"]]
    pred_box = [pred_row["x1"], pred_row["y1"], pred_row["x2"], pred_row["y2"]]

    score = iou(gt_box, pred_box)
    raw_total_iou.append(score)

    if score >= IOU_THRESHOLD:
        raw_matches += 1


gt_frames = set(gt_ball["frame"].tolist())

raw_tp = raw_matches
raw_fp = len(raw_pred_frames - gt_frames)
raw_fn = len(gt_frames) - raw_matches

raw_precision = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) > 0 else 0
raw_recall = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) > 0 else 0
raw_f1 = (
    2 * raw_precision * raw_recall / (raw_precision + raw_recall)
    if (raw_precision + raw_recall) > 0
    else 0
)

raw_mean_iou = np.mean(raw_total_iou) if raw_total_iou else 0


# ----------------------------------------------------
# Evaluate improved track as center-point accuracy
# Improved track does not have a true box, so use center distance.
# ----------------------------------------------------
improved = pd.read_csv(IMPROVED_PATH)

gt_eval = gt_ball.rename(columns={
    "x1": "gt_x1",
    "y1": "gt_y1",
    "x2": "gt_x2",
    "y2": "gt_y2"
})

merged = gt_eval.merge(improved, on="frame", how="left")

merged["gt_center_x"] = (merged["gt_x1"] + merged["gt_x2"]) / 2
merged["gt_center_y"] = (merged["gt_y1"] + merged["gt_y2"]) / 2

merged["center_error"] = np.sqrt(
    (merged["smooth_center_x"] - merged["gt_center_x"]) ** 2
    + (merged["smooth_center_y"] - merged["gt_center_y"]) ** 2
)

available = merged[merged["improved_available"] == True]

improved_mean_error = available["center_error"].mean() if len(available) > 0 else np.nan
improved_median_error = available["center_error"].median() if len(available) > 0 else np.nan
improved_available_frames = len(available)


summary = pd.DataFrame([{
    "clip_id": CLIP_ID,
    "gt_ball_frames": len(gt_ball),
    "raw_yolo_ball_frames": len(yolo_ball),
    "raw_tp_iou_0.5": raw_tp,
    "raw_fp": raw_fp,
    "raw_fn": raw_fn,
    "raw_precision_iou_0.5": raw_precision,
    "raw_recall_iou_0.5": raw_recall,
    "raw_f1_iou_0.5": raw_f1,
    "raw_mean_iou_on_matched_frames": raw_mean_iou,
    "improved_available_frames": improved_available_frames,
    "improved_mean_center_error_px": improved_mean_error,
    "improved_median_center_error_px": improved_median_error
}])

summary.to_csv(OUTPUT_PATH, index=False)

print("\nEvaluation complete.")
print(summary)
print(f"\nSaved evaluation to: {OUTPUT_PATH}")