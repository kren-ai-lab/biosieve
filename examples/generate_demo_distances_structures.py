"""Generate demo structural distance edges from biosieve_example_dataset_1000.csv.

Run from examples/raw_data_examples/
Output: struct_edges.csv
"""

import polars as pl

df = pl.read_csv("biosieve_example_dataset_1000.csv")
ids = df["id"].cast(pl.String).to_list()
N = len(ids)

rows = []
for i in range(0, N - 2, 3):
    a, b, c = ids[i], ids[i + 1], ids[i + 2]
    rows.append((a, b, 0.35))
    rows.append((a, c, 0.42))
    rows.append((b, c, 0.40))

pl.DataFrame(rows, schema=["id1", "id2", "distance"], orient="row").write_csv("struct_edges.csv")
print(f"Saved struct_edges.csv with {len(rows)} edges")
