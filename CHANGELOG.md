# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

Aligns the CLI with the shared ecosystem CLI contract.

### Added

- `--embeddings` (`.npy` or `.npz`), `--embedding-ids` and `--edges` options on `reduce`, and `--embeddings`, `--embedding-ids` and `--clusters` on `split`.
- `biosieve validate-config CONFIG --strategy NAME [--kind reduce|split] [--json]` to check a params file without reading data.
- `<output>.provenance.json` (`reduce`) and `<outdir>.provenance.json` (`split`) sidecars.
- Parquet input and output for the main dataset, chosen by file extension.
- Embedding ids as `<stem>.ids.txt` or an `ids` array inside a `.npz`.
- Tool version in the `*_report.json` files.

### Changed

- `--config` is the primary name of the params option; `--params` remains as an alias.
- Params are type-checked against the strategy fields and every problem is reported at once.
- `homology_aware` uses a temporary `work_dir` by default.
- Output tables put the id column first; the id column is always read as a string.
- The data `validate-inputs` command exits 2 on validation failures.
- Report timestamps have seconds precision.

### Breaking changes

- `embeddings_path`, `ids_path`, `edges_path` and `clusters_path` are no longer accepted in params files; use `--embeddings`, `--embedding-ids`, `--edges` and `--clusters`.
- A params block for a strategy other than `--strategy` is now an error instead of being ignored.
- Unknown parameters and wrongly typed values in params files are rejected before the run starts.
- Exit codes: config and usage errors exit 2, runtime and data errors exit 3, internal errors exit 4 (previously everything exited 1).
- The embedding ids CSV must contain the `ids_col` column (default `id`); there is no fallback to the first column.
- Duplicate embedding ids are an error.
- `homology_aware` no longer writes to `runs/mmseqs2_homology` by default.
- `embedding_cosine`, `distance_aware` and `structural_distance` no longer fall back to `embeddings.npy`, `embedding_ids.csv` or `struct_edges.csv` in the working directory.
