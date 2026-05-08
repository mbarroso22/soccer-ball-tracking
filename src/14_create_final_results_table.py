import pandas as pd

base = pd.read_csv("results/evaluation_summary.csv")
kalman = pd.read_csv("results/kalman_evaluation_summary.csv")

final = base.merge(kalman, on="clip_id", how="inner")

columns = [
    "clip_id",
    "raw_recall_iou_0.5",
    "raw_f1_iou_0.5",
    "improved_mean_center_error_px",
    "improved_median_center_error_px",
    "kalman_coverage_percent",
    "kalman_mean_error_px",
    "kalman_median_error_px",
    "kalman_error_under_50px_percent"
]

final = final[columns]

final = final.rename(columns={
    "raw_recall_iou_0.5": "YOLO Recall",
    "raw_f1_iou_0.5": "YOLO F1",
    "improved_mean_center_error_px": "Smoothed Mean Error",
    "improved_median_center_error_px": "Smoothed Median Error",
    "kalman_coverage_percent": "Kalman Coverage %",
    "kalman_mean_error_px": "Kalman Mean Error",
    "kalman_median_error_px": "Kalman Median Error",
    "kalman_error_under_50px_percent": "Kalman <50px %"
})

final.to_csv("results/final_results_table.csv", index=False)

print(final)
print("\nSaved: results/final_results_table.csv")