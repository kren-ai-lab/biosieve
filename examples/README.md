# Examples

Quick reference for all biosieve splitting and reduction strategies.

## Directory layout

```
examples/
├── configs/                    # YAML parameter files (one per strategy)
├── raw_data_examples/          # Demo data files (CSV, npy)
├── generating_random_datasets.py          # Generate base demo dataset
├── generate_demo_descriptors.py           # Add numerical descriptors columns
├── generate_demo_distances_structures.py  # Generate structural distance edges
├── prepare_cluster_mapping.py             # Extract cluster mapping CSV
├── prepare_embedding_for_reductions.py    # Convert ESM2 CSV → npy + ids CSV
├── testing_splitters.sh        # Run all split strategies
├── testing_reducers.sh         # Run all reduction strategies
└── using_sylphy_to_get_demo_embedding.sh  # Fetch ESM2 embeddings via Sylphy
```

## Setup: generate demo data

All scripts must be run from the `raw_data_examples/` directory.

```bash
cd raw_data_examples/

# 1. Base dataset (1000 peptides)
python ../generating_random_datasets.py

# 2. Dataset with descriptor columns (needed for descriptor-based strategies)
python ../generate_demo_descriptors.py

# 3. Structural distance edges (needed for structural_distance reducer)
python ../generate_demo_distances_structures.py

# 4. Cluster mapping CSV (needed for cluster_aware splitter with external mapping)
python ../prepare_cluster_mapping.py

# 5. Embeddings (needed for embedding-based strategies)
#    Option A: use Sylphy to generate real ESM2 embeddings
bash ../using_sylphy_to_get_demo_embedding.sh
python ../prepare_embedding_for_reductions.py
```

## Running the examples

```bash
cd raw_data_examples/

# All split strategies
bash ../testing_splitters.sh

# All reduction strategies
bash ../testing_reducers.sh
```

## Configs (`configs/`)

| File                                               | Strategy                                |
|----------------------------------------------------|-----------------------------------------|
| `params_split_random.yaml`                         | `random`                                |
| `params_split_random_kfold.yaml`                   | `random_kfold`                          |
| `params_split_stratified.yaml`                     | `stratified`                            |
| `params_split_stratified_kfold.yaml`               | `stratified_kfold`                      |
| `params_split_stratified_numerical.yaml`           | `stratified_numeric`                    |
| `params_split_stratified_num_kfold.yaml`           | `stratified_numeric_kfold`              |
| `params_split_group.yaml`                          | `group`                                 |
| `params_split_group_kfold.yaml`                    | `group_kfold`                           |
| `params_split_time.yaml`                           | `time`                                  |
| `params_split_distance_aware.yaml`                 | `distance_aware` (embeddings)           |
| `params_split_distance_aware_descriptors.yaml`     | `distance_aware` (descriptors)          |
| `params_split_distance_aware_embedding_kfold.yaml` | `distance_aware_kfold` (embeddings)     |
| `params_split_distance_aware_desc_kfold.yaml`      | `distance_aware_kfold` (descriptors)    |
| `params_split_homology_aware.yaml`                 | `homology_aware` (runs MMseqs2)         |
| `params_split_homology_aware_precomputed.yaml`     | `homology_aware` (precomputed clusters) |
| `params_split_cluster_aware.yaml`                  | `cluster_aware`                         |
| `params_split_cluster_aware_with_mapping.yaml`     | `cluster_aware` (external mapping file) |
| `params_reducer.yaml`                              | `embedding_cosine` reducer              |
| `params_reducer_minhash_jaccard.yaml`              | `minhash_jaccard` reducer               |

## CLI reference

```bash
# Split
biosieve split -i <input.csv> -o <outdir/> --strategy <name> --params <params.yaml>

# Reduce
biosieve reduce -i <input.csv> -o <output.csv> --strategy <name> \
  [--mapping-output map.csv] [--report-output report.json] [--params <params.yaml>]

# Override a param inline (no YAML needed)
biosieve split -i data.csv -o out/ --strategy random --set random.seed=42
```
