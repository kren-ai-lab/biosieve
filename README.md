# BioSieve

**BioSieve** is a lightweight library for:
- redundancy reduction (sequence / embedding / structure backends)
- leakage-aware dataset splitting (random, stratified, group, cluster, distance-aware, time-based)
- standardized split/reduction reporting for reproducibility

## Install (dev)
pip install -e ".[dev]"

## CLI
### Redundancy reduction
biosieve reduce --in data.csv --out data_nr.csv --strategy exact --id-col id --seq-col sequence

### Splits
biosieve split --in data_nr.csv --outdir splits --strategy stratified --label-col label --seed 13

## Philosophy
- stable contracts (same outputs across strategies)
- deterministic results with seed + stable ordering
- reporting-first (assignments + json report)
