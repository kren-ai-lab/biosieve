"""Extract a cluster mapping CSV (id -> cluster_id) from the base dataset.

Run from examples/raw_data_examples/
Output: cluster_map.csv
"""

import polars as pl

df = pl.read_csv("biosieve_example_dataset_1000.csv")
df.select(["id", "cluster_id"]).write_csv("cluster_map.csv")
print(f"Saved cluster_map.csv ({df.height} rows)")
