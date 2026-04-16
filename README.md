# BioSieve

[![Tests](https://img.shields.io/github/actions/workflow/status/kren-ai-lab/biosieve/tests.yml?style=flat-square)](https://github.com/kren-ai-lab/biosieve/actions/workflows/tests.yml)
![License](https://img.shields.io/github/license/kren-ai-lab/biosieve?style=flat-square)

BioSieve is a Python toolkit for preparing biological sequence datasets for machine learning.

It covers two main workflows:

- **Redundancy reduction** — remove near-duplicate sequences before training using sequence, embedding, descriptor, or
  structural similarity
- **Leakage-aware splitting** — partition datasets into train/val/test (or k-folds) with strategies that respect
  biological structure (homology clusters, sequence distance, groups, time)

## Installation

BioSieve supports Python 3.11+.

```bash
pip install git+https://github.com/kren-ai-lab/biosieve.git
```

Install optional extras as needed:

- `minhash` for approximate Jaccard-based deduplication (`minhash_jaccard` strategy)
- `faiss` for GPU-accelerated embedding similarity search (`embedding_cosine` strategy)

```bash
pip install 'biosieve[minhash] @ git+https://github.com/kren-ai-lab/biosieve.git'
pip install 'biosieve[faiss] @ git+https://github.com/kren-ai-lab/biosieve.git'
```

The `mmseqs2` reducer and `homology_aware` splitter require the [MMseqs2](https://github.com/soedinglab/MMseqs2) binary
to be available in `PATH`.

> [!TIP]
> You can install MMSeqs2 easlity with [pixi](https://pixi.prefix.dev/latest/): `pixi global install -c bioconda -c conda-forge mmseqs2`. 

## Quick Start

### Redundancy reduction

Remove near-duplicate sequences using k-mer Jaccard similarity:

```bash
biosieve reduce \
  -i dataset.csv \
  -o dataset_nr.csv \
  --strategy kmer_jaccard \
  --mapping-output mapping.csv \
  --report-output report.json
```

Pass parameters via a YAML file:

```bash
biosieve reduce \
  -i dataset.csv \
  -o dataset_nr.csv \
  --strategy kmer_jaccard \
  --params params.yaml
```

```yaml
# params.yaml
kmer_jaccard:
  threshold: 0.8
  k: 5
```

Override a single parameter inline without a file:

```bash
biosieve reduce -i dataset.csv -o out.csv --strategy kmer_jaccard --set kmer_jaccard.threshold=0.9
```

### Dataset splitting

Split with a leakage-aware strategy:

```bash
biosieve split \
  -i dataset_nr.csv \
  -o splits/ \
  --strategy homology_aware \
  --params params.yaml
```

```yaml
# params.yaml
homology_aware:
  mode: precomputed
  clusters_path: clusters.csv
  member_col: id
  cluster_col: cluster_id
  test_size: 0.2
```

## Strategies

### Redundancy reduction

| Strategy               | Description                                           | Extra needed                 |
|------------------------|-------------------------------------------------------|------------------------------|
| `exact`                | Remove exact sequence duplicates                      | —                            |
| `identity_greedy`      | Greedy reduction by sequence identity                 | —                            |
| `kmer_jaccard`         | Greedy reduction by k-mer Jaccard similarity          | —                            |
| `minhash_jaccard`      | Approximate k-mer Jaccard via MinHash LSH (fast)      | `biosieve[minhash]`          |
| `embedding_cosine`     | Cosine similarity on precomputed embeddings           | `biosieve[faiss]` (optional) |
| `descriptor_euclidean` | Euclidean distance on numeric descriptor columns      | —                            |
| `structural_distance`  | Graph-based reduction on precomputed structural edges | —                            |
| `mmseqs2`              | Homology clustering via MMseqs2                       | `mmseqs` binary              |

### Splitting

| Strategy             | Description                                                        |
|----------------------|--------------------------------------------------------------------|
| `random`             | Random train/val/test split                                        |
| `stratified`         | Stratified by a categorical label column                           |
| `stratified_numeric` | Stratified by a numeric label column (binned)                      |
| `group`              | No group appears in more than one split                            |
| `time`               | Chronological split by a time column                               |
| `cluster_aware`      | Group split using a precomputed cluster column                     |
| `distance_aware`     | Test set selected as farthest points in embedding/descriptor space |
| `homology_aware`     | Group split derived from MMseqs2 clusters or precomputed clusters  |

All strategies also support k-fold variants (`random_kfold`, `stratified_kfold`, `group_kfold`,
`stratified_numeric_kfold`, `distance_aware_kfold`).

## Outputs

Every run produces consistent artefacts:

- Reduced or split CSVs
- Mapping CSV (`removed_id`, `representative_id`, `cluster_id`, `score`) for reduction runs
- JSON report with strategy name, effective parameters, and reduction/split statistics

## Learn More

- [examples/README.md](examples/README.md) for runnable scripts and config files

## License

**GPL-3.0-or-later**. See [LICENSE](LICENSE).

## Acknowledgements

Built on top of scikit-learn, pandas, NumPy, and optionally datasketch, FAISS, and MMseqs2.

Developed by **KREN AI Lab** at Universidad de Magallanes, Chile.
