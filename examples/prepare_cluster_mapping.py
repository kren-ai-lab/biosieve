"""Extract a cluster mapping CSV (id -> cluster_id) from the base dataset.

Run from examples/raw_data_examples/
Output: cluster_map.csv
"""

import pandas as pd

df = pd.read_csv("biosieve_example_dataset_1000.csv")
df[["id", "cluster_id"]].to_csv("cluster_map.csv", index=False)
print(f"Saved cluster_map.csv ({len(df)} rows)")
