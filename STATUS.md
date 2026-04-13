# BioSieve — Project Status

BioSieve is a toolkit for **redundancy reduction** and **dataset splitting** designed for biological sequence datasets (
and optionally embeddings/descriptors/structures). The goal is to provide **leakage-aware, reproducible** workflows via
a stable CLI + config system.

---

## 1) Scope (What BioSieve is / is not)

### In scope

- Redundancy reduction strategies (sequence, embedding, descriptor, structural).
- Splitting strategies (random, stratified, group-aware, time-based, distance-aware, homology-aware).
- Reproducible execution via:
    - CLI subcommands (`reduce`, `split`)
    - Param files (`--params` YAML/JSON) + overrides (`--set`)
    - Reports and mappings as artefacts.

### Out of scope (for now[]())

- End-to-end feature extraction (e.g., generating embeddings).
- Full homology pipelines beyond clustering (BioSieve can run MMseqs2 in v0.1; deeper pipelines are external).
- Model training/evaluation (belongs to Eris / downstream libs).

---

## 2) Current release posture

- Current development stage: **pre-release / v0.1.x**
- Stability target:
    - CLI flags + report schema: **stable by v0.2.0**
    - internal APIs: may change until v0.2.0

---

## 3) CLI surface

### Implemented

- `biosieve reduce` ✅
- `biosieve split` ✅
- Global config system:
    - `--params <file.{yaml,json}>` ✅
    - `--set strategy.key=value` ✅

### Planned

- `biosieve validate` (input + features consistency checks) ⏳
- `biosieve info` (list available strategies + accepted params) ⏳

---

## 4) Strategies matrix

### 4.1 Redundancy reduction strategies (`biosieve reduce`)

| Strategy               | Status | Input requirements                | Notes                            |
|------------------------|-------:|-----------------------------------|----------------------------------|
| `exact`                |      ✅ | `id`, (optional `sequence`)       | exact duplicate removal          |
| `identity_greedy`      |      ✅ | `sequence`                        | greedy identity-based reduction  |
| `kmer_jaccard`         |      ✅ | `sequence`                        | optional dependency `datasketch` |
| `mmseqs2`              |      ✅ | `sequence`, `mmseqs2` binary      | homology clustering / filtering  |
| `embedding_cosine`     |      ✅ | embeddings (`.npy` + ids mapping) | FAISS optional; sklearn fallback |
| `descriptor_euclidean` |      ✅ | numeric descriptor columns        | optional standardization         |
| `structural_distance`  |      ✅ | precomputed structural edges      | consumes edge list / distances   |

**Artefacts produced:**

- reduced dataset CSV ✅
- mapping CSV (removed → representative) ✅
- JSON report ✅

---

### 4.2 Splitting strategies (`biosieve split`)

| Strategy         | Status | Input requirements                          | Notes                                                   |
|------------------|-------:|---------------------------------------------|---------------------------------------------------------|
| `random`         |      ✅ | `id`                                        | deterministic with seed                                 |
| `stratified`     |      ✅ | `label` column (categorical)                | sklearn required                                        |
| `group`          |      ✅ | `group_col`                                 | strict no-group overlap                                 |
| `time`           |      ✅ | `time_col`                                  | chronological split                                     |
| `distance_aware` |      ✅ | embeddings or descriptors                   | "test most different" (v0.1 centroid-farthest)          |
| `homology_aware` |      ✅ | sequences + mmseqs2 OR precomputed clusters | clusters → group split                                  |
| `cluster`        |     ⚠️ | (exists in repo)                            | not fully integrated / not registered (decision needed) |

**Artefacts produced:**

- `train.csv` ✅
- `test.csv` ✅
- `val.csv` ✅ (only if `val_size > 0`)
- split report JSON ✅

---

## 5) Configuration contract

### Params file format (YAML/JSON)

Root object must be:

```yaml
strategy_name:
  param1: value
  param2: value
