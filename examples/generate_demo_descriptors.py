"""Add 32 random numerical descriptors to the base dataset.

Run from examples/raw_data_examples/
Output: biosieve_example_dataset_1000_desc.csv
"""

import numpy as np
import pandas as pd

df = pd.read_csv("biosieve_example_dataset_1000.csv")

rng = np.random.default_rng(13)
for i in range(32):
    df[f"desc_{i:03d}"] = rng.normal(size=len(df)).astype("float32")

df.to_csv("biosieve_example_dataset_1000_desc.csv", index=False)
print(f"Saved biosieve_example_dataset_1000_desc.csv ({len(df)} rows, {len(df.columns)} cols)")
