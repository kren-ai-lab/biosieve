"""Execution runner for reduction workflows."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, cast

import biosieve
from biosieve._provenance import utc_now, write_provenance
from biosieve.core.common import (
    JSONValue,
    check_paths,
    effective_params,
    ensure_parent,
    provenance_inputs,
    read_table,
    safe_jsonable,
    utc_timestamp,
    validate_unique_ids,
    write_json,
    write_table,
)
from biosieve.reduction.common import empty_mapping_df
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.core.registry import StrategyRegistry
    from biosieve.reduction.base import Reducer

log = get_logger(__name__)


def run_reduce(
    in_path: str,
    out_path: str,
    strategy: str,
    registry: StrategyRegistry,
    *,
    cols: Columns | None = None,
    map_path: str | None = None,
    report_path: str | None = None,
    strategy_params: dict[str, object] | None = None,
    read_csv_kwargs: dict[str, object] | None = None,
    paths: dict[str, str] | None = None,
) -> None:
    r"""Execute redundancy reduction and export artefacts to disk.

    Artefact contract:
    - Reduced dataset: `out_path` (CSV or Parquet, by extension), `id` column first
    - Provenance sidecar: `<out_path>.provenance.json`
    - Optional mapping: `map_path` (CSV or Parquet)
      If `map_path` is provided and the reducer returns no mapping, an empty mapping
      CSV is written with stable columns:
        ["removed_id", "representative_id", "cluster_id", "score"]
    - Optional report: `report_path` (JSON)

    Args:
        in_path: Input CSV or Parquet path.
        out_path: Output CSV path for the non-redundant dataset.
        strategy: Reducer strategy name (e.g., "mmseqs2", "embedding_cosine", "descriptor_euclidean").
        registry: StrategyRegistry holding reducer classes in `registry.reducers`.
        cols: Columns specification. If None, defaults to Columns(id_col="id", seq_col="sequence").
        map_path: Optional mapping CSV path.
        report_path: Optional JSON report path.
        strategy_params: Params used to instantiate the reducer class (unknown keys -> error).
        read_csv_kwargs: Optional kwargs passed to polars.read_csv.
        paths: Input files for the strategy, keyed by strategy field (e.g. `embeddings_path`).

    Raises:
        ValueError: If the reducer strategy name is unknown, the id column is missing, or ids
        are not unique.
        FileNotFoundError: If `in_path` does not exist (raised by polars).
        RuntimeError: If the reducer fails internally (propagated from the reducer implementation).

    Notes:
        - Some reducers may not require sequences. Therefore, missing `cols.seq_col` is not
        a hard error at runner level. Reducers should validate required columns themselves.
        - Reports attempt to be JSON-serializable and stable for downstream use (MCS-friendly).

    Examples:
        Minimal usage:

    >>> biosieve reduce \\
    ...   --in dataset.csv \\
    ...   --out data_nr.csv \\
    ...   --strategy embedding_cosine \\
    ...   --params params.yaml \\
    ...   --map mapping.csv \\
    ...   --report reduction.json

    """
    t0 = time.time()
    created_at = utc_now()
    log.info(
        "reduce:start | strategy=%s | in=%s | out=%s | map=%s | report=%s",
        strategy,
        in_path,
        out_path,
        map_path,
        report_path,
    )

    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    paths = paths or {}

    if strategy not in registry.reducers:
        available = sorted(registry.reducers.keys())
        msg = f"Unknown reducer strategy '{strategy}'. Available: {available}"
        raise ValueError(msg)

    ensure_parent(out_path)
    ensure_parent(map_path)
    ensure_parent(report_path)

    check_paths(registry.get_reducer_class(strategy), paths)
    df = read_table(in_path, cols.id_col, read_csv_kwargs)

    log.info("reduce:input | n_rows=%d | n_cols=%d", df.height, len(df.columns))

    validate_unique_ids(df, cols.id_col)

    # Soft check for sequence column (reducers decide if required)
    # We do not raise here to keep descriptor/structural reducers usable.
    # Reducers SHOULD raise if they require sequence and it is missing.

    reducer = cast("Reducer", registry.create_reducer(strategy, {**strategy_params, **paths}))

    res = reducer.run(df, cols)

    # --- write outputs ---
    write_table(res.df, out_path, cols.id_col)

    mapping_df = res.mapping if res.mapping is not None else empty_mapping_df()
    if map_path is not None:
        write_table(mapping_df, map_path, "removed_id")

    # --- report ---
    if report_path is not None:
        n_in = df.height
        n_out = res.df.height
        n_removed = int(n_in - n_out)

        report: dict[str, JSONValue] = {
            "schema_version": "0.1",
            "tool": {"name": "biosieve", "version": biosieve.__version__},
            "timestamp": utc_timestamp(),
            "in_path": str(in_path),
            "out_path": str(out_path),
            "map_path": str(map_path) if map_path else None,
            "strategy": str(strategy),
            "strategy_params": safe_jsonable(strategy_params),
            "effective_params": safe_jsonable(res.params),
            "summary": {
                "n_in": n_in,
                "n_out": n_out,
                "n_removed": n_removed,
                "fraction_removed": float(n_removed / n_in) if n_in else 0.0,
                "n_mapped_removed": mapping_df.height,
            },
            "stats": safe_jsonable(res.stats),
            "columns": safe_jsonable(cols),
        }
        write_json(report_path, report)

    write_provenance(
        Path(out_path),
        tool="biosieve",
        version=biosieve.__version__,
        command="reduce",
        created_at=created_at,
        parameters={
            "strategy": strategy,
            "params": effective_params(reducer),
            "columns": asdict(cols),
            "read_options": read_csv_kwargs or {},
        },
        inputs=provenance_inputs(in_path, paths),
        packages=("biosieve", "polars", "numpy"),
        output_info={"n_rows": res.df.height, "n_cols": len(res.df.columns)},
        extensions={
            "strategy": strategy,
            "summary": {"n_in": df.height, "n_out": res.df.height, "n_removed": df.height - res.df.height},
            "stats": safe_jsonable(res.stats),
        },
    )

    log.info(
        "reduce:result | n_kept=%d | n_removed=%d | ratio=%.4f",
        len(res.df),
        len(mapping_df),
        (len(res.df) / len(df)) if len(df) else 0.0,
    )

    elapsed = time.time() - t0
    log.info("reduce:end | seconds=%.3f | out=%s", elapsed, out_path)
