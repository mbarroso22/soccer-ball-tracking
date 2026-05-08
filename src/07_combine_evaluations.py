import pandas as pd
from pathlib import Path

files = sorted(Path("results").glob("evaluation_SNMOT-*.csv"))

dfs = [pd.read_csv(f) for f in files]

combined = pd.concat(dfs, ignore_index=True)

combined.to_csv("results/evaluation_summary.csv", index=False)

print(combined)
print("\nSaved: results/evaluation_summary.csv")