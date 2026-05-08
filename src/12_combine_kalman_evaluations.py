import pandas as pd
from pathlib import Path

files = sorted(Path("results").glob("kalman_evaluation_SNMOT-*.csv"))

combined = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
combined.to_csv("results/kalman_evaluation_summary.csv", index=False)

print(combined)
print("\nSaved: results/kalman_evaluation_summary.csv")