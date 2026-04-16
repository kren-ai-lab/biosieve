"""Generate demo structural distance edges from biosieve_example_dataset_1000.csv.

Run from examples/raw_data_examples/
Output: struct_edges.csv
"""

import numpy as np
import pandas as pd

df = pd.read_csv("biosieve_example_dataset_1000.csv")
ids = df["id"].astype(str).tolist()
N = len(ids)

rows = []
for i in range(0, N - 2, 3):
    a, b, c = ids[i], ids[i + 1], ids[i + 2]
    rows.append((a, b, 0.35))
    rows.append((a, c, 0.42))
    rows.append((b, c, 0.40))

pd.DataFrame(rows, columns=["id1", "id2", "distance"]).to_csv("struct_edges.csv", index=False)
print(f"Saved struct_edges.csv with {len(rows)} edges")
