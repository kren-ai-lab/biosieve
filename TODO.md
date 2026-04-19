
# BioSieve — Roadmap

## Reducers

### Representative selection policy (`rep_policy`)

Currently all greedy reducers pick the **first** sequence as representative (deterministic by sorted id).
Configurable policy options:

- `first` (current default)
- `longest_sequence`
- `best_quality` (requires a quality column)
- `earliest_time` (requires a time column)
- `max_label_value`

---

## Splitters

### `nested_cv`

Outer split (leakage-aware) + inner CV folds. Useful for proper model selection on biological data.

### `structural_aware`

Split based on precomputed structural distances (`id1, id2, dist` edge list):

1. Graph clustering: threshold → clusters → group split
2. Connected components: threshold → components → split by component

### Hybrid strategies

- **`homology_then_distance`** — cluster by homology first (no leakage), then distance-aware selection within train/val
- **`homology_and_stratified_numeric`** — homology clusters → bin by numeric label (cluster median) → greedy cluster assignment preserving bin distribution
- **`time_and_leakage_constraints`** — temporal split (train earlier / test later) + guardrail: if a cluster appears in test, no cluster member leaks into train

---

## Interop

- Sylphy embedding export format: validate `embeddings.npy` + `embedding_ids.csv` + `meta.json` layout in `biosieve validate`
- Eris-compatible output naming: `splits/train.csv`, `splits/test.csv`, `splits/val.csv`, `folds/fold_*/`
