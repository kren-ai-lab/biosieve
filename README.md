# BioSieve

**BioSieve** is a lightweight library for:
- redundancy reduction (sequence / embedding / structure backends)
- leakage-aware dataset splitting (random, stratified, group, cluster, distance-aware, time-based)
- standardized split/reduction reporting for reproducibility

## Install (dev)
pip install -e ".[dev]"

## CLI
### Redundancy reduction
biosieve reduce --input-data data.csv --output data_nr.csv --strategy exact --id-column id --sequence-column sequence

### Splits
biosieve split --input-data data_nr.csv --output-dir splits --strategy stratified

## Philosophy
- stable contracts (same outputs across strategies)
- deterministic results with seed + stable ordering
- reporting-first (assignments + json report)
