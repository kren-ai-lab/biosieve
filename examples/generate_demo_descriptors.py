"""Add 32 random numerical descriptors to the base dataset.

Run from examples/raw_data_examples/
Output: biosieve_example_dataset_1000_desc.csv
"""

import numpy as np
import polars as pl

df = pl.read_csv("biosieve_example_dataset_1000.csv")

rng = np.random.default_rng(13)
for i in range(32):
    df = df.with_columns(pl.Series(f"desc_{i:03d}", rng.normal(size=df.height).astype("float32")))

df.write_csv("biosieve_example_dataset_1000_desc.csv")
print(f"Saved biosieve_example_dataset_1000_desc.csv ({df.height} rows, {len(df.columns)} cols)")
