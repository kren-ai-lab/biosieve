"""Execution runner for reduction workflows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import pandas as pd

from biosieve.core.factory import instantiate_strategy
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.core.registry import StrategyRegistry
    from biosieve.reduction.base import Reducer

log = get_logger(__name__)

JSONScalar: TypeAlias = str | int | float | bool | None  # noqa: UP040
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]  # noqa: UP040


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp with a trailing Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ensure_parent(path: str | None) -> None:
    """Create parent directory for a file path if needed."""
    if not path:
        return
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _safe_jsonable(x: object) -> JSONValue:
    """Convert objects into JSON-serializable representations (best-effort).

    Args:
        x: Arbitrary Python object.

    Returns:
        JSON-serializable version of `x`.

    Notes:
        - Dataclasses are converted via `asdict`.
        - Unknown objects are converted to `str(x)`.

    """
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, (list, tuple)):
        return [_safe_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _safe_jsonable(v) for k, v in x.items()}
    if is_dataclass(x) and not isinstance(x, type):
        return _safe_jsonable(asdict(x))
    return str(x)


def _validate_unique_ids(df: pd.DataFrame, id_col: str) -> None:
    """Fail-fast check: BioSieve expects 1 row = 1 unique id."""
    if id_col not in df.columns:
        msg = f"Missing id column '{id_col}' in input data. Columns: {df.columns.tolist()}"
        raise ValueError(msg)
    n_in = len(df)
    unique_ids = df[id_col].astype(str).nunique()
    if unique_ids != n_in:
        msg = (
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{id_col}'."
        )
        raise ValueError(msg)


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
) -> None:
    r"""Execute redundancy reduction and export artefacts to disk.

    Artefact contract:
    - Reduced dataset: `out_path` (CSV)
    - Optional mapping: `map_path` (CSV)
      If `map_path` is provided and the reducer returns no mapping, an empty mapping
      CSV is written with stable columns:
        ["removed_id", "representative_id", "cluster_id", "score"]
    - Optional report: `report_path` (JSON)

    Args:
        in_path: Input CSV path.
        out_path: Output CSV path for the non-redundant dataset.
        strategy: Reducer strategy name (e.g., "mmseqs2", "embedding_cosine", "descriptor_euclidean").
        registry: StrategyRegistry holding reducer classes in `registry.reducers`.
        cols: Columns specification. If None, defaults to Columns(id_col="id", seq_col="sequence").
        map_path: Optional mapping CSV path.
        report_path: Optional JSON report path.
        strategy_params: Params used to instantiate the reducer class (unknown keys -> error).
        read_csv_kwargs: Optional kwargs passed to pandas.read_csv.

    Raises:
        ValueError: If the reducer strategy name is unknown, the id column is missing, or ids
        are not unique.
        FileNotFoundError: If `in_path` does not exist (raised by pandas).
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
    read_csv_kwargs = read_csv_kwargs or {}

    if strategy not in registry.reducers:
        available = sorted(registry.reducers.keys())
        msg = f"Unknown reducer strategy '{strategy}'. Available: {available}"
        raise ValueError(msg)

    _ensure_parent(out_path)
    _ensure_parent(map_path)
    _ensure_parent(report_path)

    df = pd.read_csv(in_path, **cast("dict[str, Any]", read_csv_kwargs))

    log.info("reduce:input | n_rows=%d | n_cols=%d", len(df), len(df.columns))

    _validate_unique_ids(df, cols.id_col)

    # Soft check for sequence column (reducers decide if required)
    # We do not raise here to keep descriptor/structural reducers usable.
    # Reducers SHOULD raise if they require sequence and it is missing.

    reducer_cls = registry.get_reducer_class(strategy)
    reducer = cast("Reducer", instantiate_strategy(reducer_cls, strategy_params))

    res = reducer.run(df, cols)

    # --- write outputs ---
    res.df.to_csv(out_path, index=False)

    if map_path is not None:
        if res.mapping is None:
            pd.DataFrame(columns=["removed_id", "representative_id", "cluster_id", "score"]).to_csv(
                map_path, index=False
            )
        else:
            res.mapping.to_csv(map_path, index=False)

    # --- report ---
    if report_path is not None:
        n_in = len(df)
        n_out = len(res.df)
        n_removed = int(n_in - n_out)

        report: dict[str, JSONValue] = {
            "schema_version": "0.1",
            "timestamp": _utc_timestamp(),
            "in_path": str(in_path),
            "out_path": str(out_path),
            "map_path": str(map_path) if map_path else None,
            "strategy": str(strategy),
            "strategy_params": _safe_jsonable(strategy_params),
            "effective_params": _safe_jsonable(getattr(res, "params", {})),
            "stats": {
                "n_in": n_in,
                "n_out": n_out,
                "n_removed": n_removed,
                "fraction_removed": float(n_removed / n_in) if n_in else 0.0,
                "n_mapped_removed": len(res.mapping) if res.mapping is not None else 0,
            },
            "columns": _safe_jsonable(cols),
        }

        if getattr(res, "stats", None) is not None:
            report["reducer_stats"] = _safe_jsonable(res.stats)

        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")

    log.info(
        "reduce:result | n_kept=%d | n_removed=%d | ratio=%.4f",
        len(res.df),
        len(res.mapping) if res.mapping is not None else 0,
        (len(res.df) / len(df)) if len(df) else 0.0,
    )

    elapsed = time.time() - t0
    log.info("reduce:end | seconds=%.3f | out=%s", elapsed, out_path)
