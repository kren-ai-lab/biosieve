
## 0) Fundaciones de ingeniería

### 0.1 Docstrings y documentación interna

* **Docstrings estilo NumPy** en:

  * todos los splitters (`splitting/*`)
  * todos los reducers (`reduction/*`)
  * runners (`core/runner.py`, `core/split_runner.py`)
  * backends (`reduction/backends/*`)
  * params (`io/params.py`)
* Docstrings deben incluir siempre:

  * `Parameters` con defaults
  * `Returns` (incluyendo schema de report/mapping)
  * `Raises` con errores típicos (missing columns, missing files, etc.)
  * `Notes` sobre leakage/coverage y decisiones de diseño
  * `Examples` (1 comando CLI real)

### 0.2 Tipado y contratos

* Type hints completos y consistentes:

  * `SplitResult`, `ReductionResult`, `Columns`
  * `StrategyRegistry` (tipar `Dict[str, type]`)
* Validación de params:

  * unknown keys → error (evitar typos silenciosos)
  * validación de rangos (`test_size`, thresholds, etc.)
* Contract de artefactos:

  * naming estable para outputs (`train.csv`, `test.csv`, `val.csv`, `split_report.json`, etc.)

### 0.3 Imports eficientes (`__init__` + lazy loading)

* Evitar imports pesados al `import biosieve`:

  * `mmseqs2`, `sklearn`, `datasketch`, `faiss` deben ser **lazy** (import dentro de funciones)
* `__init__.py` minimalistas:

  * exportar solo API liviana
  * no importar “todas las estrategias” si eso dispara deps
* Agregar `biosieve info` para descubrir estrategias sin cargar todo (vía introspección del registry, sin instanciar).

### 0.4 Logging serio (nivel equipo)

* Estándar: `biosieve.utils.logging.get_logger(name)`
* CLI flags:

  * `--log-level {DEBUG,INFO,WARNING,ERROR}`
  * `--quiet`
  * `--log-file`
* Logs estructurados en runners:

  * start/end, tiempos, n_rows, coverage, paths
  * warnings: missing embeddings/ids, bins pequeños, etc.
* Cada estrategia debe loguear:

  * parámetros efectivos
  * decisiones (fallbacks, reducción de bins, etc.)
  * stats clave

### 0.5 Tests + CI

* Unit tests por estrategia (mínimo):

  * 1 test “happy path”
  * 1 test “missing column/file”
  * 1 test “leakage=0” (group/homology/cluster)
* CLI integration tests (subprocess):

  * `biosieve reduce ...`
  * `biosieve split ...`
* GitHub Actions:

  * Python 3.10/3.11/3.12
  * ruff + mypy (opcional) + pytest
* Fixtures: dataset pequeño + embeddings dummy + edges dummy

---

## 1) UX “Sylphy-level”

### 1.1 Comandos “meta”

* `biosieve info`

  * lista estrategias `reduce` y `split`
  * muestra params + defaults (autogenerado desde dataclasses)
  * ejemplo YAML por estrategia
* `biosieve validate`

  * input checks (ids únicos, columnas requeridas)
  * embeddings alignment: ids ↔ filas de `.npy`
  * descriptors: NaNs, no-numéricos, escala recomendada
  * structural edges: ids presentes, simetría opcional, rangos
  * mmseqs2 binary disponible (si se usa)
* `biosieve doctor` (opcional)

  * imprime versiones, rutas de binarios, deps opcionales disponibles

### 1.2 Examples completos

* `examples/`

  * `dataset.csv` (con id, sequence, label_cls, label_reg, time, group)
  * `embeddings.npy` + `embedding_ids.csv`
  * `struct_edges.csv` (o parquet)
  * `clusters.tsv` (mmseqs2 demo)
* `README` con:

  * 5 comandos copy/paste (reduce y split)
  * tabla “qué estrategia usar cuándo”
  * troubleshooting de errores típicos

### 1.3 Artefactos reproducibles

* Guardar `params_effective.json` dentro de `outdir`
* Guardar fingerprint del input (hash ligero de file + size + mtime)
* Reportes con schema estable

---

## 2) Splits: estrategias base (deben quedar 100% pulidas)

### 2.1 Ya existentes (pulir)

* `random` (ya)
* `stratified` (clasificación)
* `group` (por taxid/family/subject)
* `time` (chronological)
* `distance_aware` (centroid-farthest)
* `cluster_aware` (wrapper group sobre cluster_id)
* `homology_aware` (mmseqs2/precomputed)

### 2.2 Pendientes “core”

* **`nested_cv`** (opcional, pero vendible):

  * outer split leakage-aware + inner CV

---

## 3) Splits basados en estructuras (lo que pediste explícito)

### 3.1 Structural-aware split (precomputed)

Estrategia: `structural_aware`

* input:

  * `struct_edges.csv` con distancias o similitudes (id1,id2,dist)
* objetivo:

  * evitar leakage estructural (poses similares, folds similares)
* variantes:

  1. **Graph clustering**: construir grafo (edges con dist < threshold) → clusters → group split
  2. **Connected components**: threshold → componentes → split por componente
  3. **Farthest test**: seleccionar nodos/cluster más lejos (requiere embedding/graph distance)

### 3.2 Structure-time hybrid (interesante)

* `struct_time`:

  * train en estructuras “antiguas” + test en estructuras “nuevas” si tienes timestamp de deposición/PDB date
  * útil si tu dataset crece en el tiempo

---

## 4) Redundancy reduction: completar “portfolio” y robustecer

### 4.1 Reducers ya presentes (pulir)

* exact / identity / kmer / mmseqs2 / embedding / descriptor / structural
* mejoras:

  * `rep_policy` configurable:

    * `first`, `longest_sequence`, `best_quality`, `earliest_time`, `max_label_value`, etc.
  * `keep_columns` / `drop_columns` stable
  * `report` con coverage y stats por etapa

### 4.2 Reducers pendientes “estructura”

* `structural_threshold_dedup`:

  * elimina si dist < threshold (graph-based representative selection)
* `structural_cluster_dedup`:

  * cluster por estructura y tomar reps por cluster

---

## 5) Estrategias híbridas (las más interesantes para BioSieve)

### 5.1 Split híbrido por leakage (multi-constraint)

**Objetivo**: “no leakage” con múltiples señales:

* homology clusters (mmseqs2)
* structural clusters
* group labels (taxid/subject)
* (opcional) time ordering

Estrategias híbridas propuestas:

1. **`homology_then_distance`**

   * primero split por clusters homólogos (no leakage)
   * dentro del train (o val) hacer distance-aware para seleccionar hard cases

2. **`distance_then_cluster`**

   * seleccionar test “farthest” en embedding
   * luego expandir test a clusters completos (cluster-aware) para no mezclar clusters

3. **`homology_and_stratified_numeric`**

   * clusterizar por homología
   * binning numérico por cluster (median)
   * asignar clusters a splits preservando bins (greedy balancing)

4. **`structure_and_homology`**

   * construir clusters por estructura
   * merge con clusters por homología (union-find de constraints)
   * group split sobre el cluster final

5. **`time_and_leakage_constraints`**

   * split temporal (train earlier, test later)
   * pero además: si un cluster aparece en test, ningún miembro del cluster aparece en train (o se filtra)
   * es “time-first with leakage guardrails”

### 5.2 Validación “intermedia” (tu idea de custom policies)

* `distance_policy` para seleccionar **test** y **val**:

  * `test_policy=farthest`
  * `val_policy=closest_to_centroid` (val más “fácil”/representativa)
  * `val_policy=midpoint` (val intermedia)
* Reportar: distancias promedio por split (train/val/test) para justificar.

---

## 6) Interoperabilidad con tu stack (Sylphy/Eris/MCS)

### 6.1 Sylphy embeddings

* Loader oficial “sylphy-export format”

  * `embeddings.npy`
  * `embedding_ids.csv`
  * `meta.json` (modelo, layer, pooling)
* `biosieve validate` debe entender este layout

### 6.2 MCS artefacts

* outputs en `outdir` deben ser “artefactos MCS-friendly”:

  * config snapshot
  * report schema version
  * deterministic filenames

### 6.3 Eris training

* producir splits con naming consistente para que Eris los tome sin glue:

  * `splits/train.csv`, `splits/test.csv`, `splits/val.csv`
  * o `folds/fold_*/...`

---

## 7) Prioridad recomendada para terminar “nivel Sylphy”

### Fase A (producto usable ya)

3. `biosieve info`
4. `biosieve validate`
5. logging end-to-end

### Fase B (paper-mode / leakage pro)

6. `homology + stratified_numeric` (cluster-level balancing)
7. `structural_aware split` (graph clustering)
8. híbridas (`distance_then_cluster`, `structure_and_homology`)

### Fase C (mantenimiento)

9. tests + CI + docs + examples pulidos
10. schema versioning de reports