from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from biosieve.core.factory import instantiate_strategy
from biosieve.core.registry import StrategyRegistry
from biosieve.types import Columns


def _ensure_parent(path: Optional[str]) -> None:
    """Create parent directory for a file path if needed."""
    if not path:
        return
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _safe_jsonable(x: Any) -> Any:
    """
    Convert objects into JSON-serializable representations (best-effort).

    Parameters
    ----------
    x:
        Arbitrary Python object.

    Returns
    -------
    Any
        JSON-serializable version of `x`.

    Notes
    -----
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
    if is_dataclass(x):
        return _safe_jsonable(asdict(x))
    return str(x)


def _validate_unique_ids(df: pd.DataFrame, id_col: str) -> None:
    """Fail-fast check: BioSieve expects 1 row = 1 unique id."""
    if id_col not in df.columns:
        raise ValueError(f"Missing id column '{id_col}' in input data. Columns: {df.columns.tolist()}")
    n_in = len(df)
    unique_ids = df[id_col].astype(str).nunique()
    if unique_ids != n_in:
        raise ValueError(
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{id_col}'."
        )


def run_reduce(
    in_path: str,
    out_path: str,
    strategy: str,
    registry: StrategyRegistry,
    *,
    cols: Optional[Columns] = None,
    map_path: Optional[str] = None,
    report_path: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Execute redundancy reduction and export artefacts to disk.

    Artefact contract
    -----------------
    - Reduced dataset: `out_path` (CSV)
    - Optional mapping: `map_path` (CSV)
      If `map_path` is provided and the reducer returns no mapping, an empty mapping
      CSV is written with stable columns:
        ["removed_id", "representative_id", "cluster_id", "score"]
    - Optional report: `report_path` (JSON)

    Parameters
    ----------
    in_path:
        Input CSV path.
    out_path:
        Output CSV path for the non-redundant dataset.
    strategy:
        Reducer strategy name (e.g., "mmseqs2", "embedding_cosine", "descriptor_euclidean").
    registry:
        StrategyRegistry holding reducer classes in `registry.reducers`.
    cols:
        Columns specification. If None, defaults to Columns(id_col="id", seq_col="sequence").
    map_path:
        Optional mapping CSV path.
    report_path:
        Optional JSON report path.
    strategy_params:
        Params used to instantiate the reducer class (unknown keys -> error).
    read_csv_kwargs:
        Optional kwargs passed to pandas.read_csv.

    Raises
    ------
    ValueError
        If the reducer strategy name is unknown, the id column is missing, or ids
        are not unique.
    FileNotFoundError
        If `in_path` does not exist (raised by pandas).
    RuntimeError
        If the reducer fails internally (propagated from the reducer implementation).

    Notes
    -----
    - Some reducers may not require sequences. Therefore, missing `cols.seq_col` is not
      a hard error at runner level. Reducers should validate required columns themselves.
    - Reports attempt to be JSON-serializable and stable for downstream use (MCS-friendly).

    Examples
    --------
    Minimal usage:

    >>> biosieve reduce \\
    ...   --in dataset.csv \\
    ...   --out data_nr.csv \\
    ...   --strategy embedding_cosine \\
    ...   --params params.yaml \\
    ...   --map mapping.csv \\
    ...   --report reduction.json
    """
    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    read_csv_kwargs = read_csv_kwargs or {}

    if strategy not in registry.reducers:
        available = sorted(list(registry.reducers.keys()))
        raise ValueError(f"Unknown reducer strategy '{strategy}'. Available: {available}")

    _ensure_parent(out_path)
    _ensure_parent(map_path)
    _ensure_parent(report_path)

    df = pd.read_csv(in_path, **read_csv_kwargs)

    _validate_unique_ids(df, cols.id_col)

    # Soft check for sequence column (reducers decide if required)
    # We do not raise here to keep descriptor/structural reducers usable.
    # Reducers SHOULD raise if they require sequence and it is missing.

    reducer_cls = registry.reducers[strategy]
    reducer = instantiate_strategy(reducer_cls, strategy_params)

    res = reducer.run(df, cols)

    # --- write outputs ---
    res.df.to_csv(out_path, index=False)

    if map_path is not None:
        if res.mapping is None:
            pd.DataFrame(
                columns=["removed_id", "representative_id", "cluster_id", "score"]
            ).to_csv(map_path, index=False)
        else:
            res.mapping.to_csv(map_path, index=False)

    # --- report ---
    if report_path is not None:
        n_in = int(len(df))
        n_out = int(len(res.df))
        n_removed = int(n_in - n_out)

        report: Dict[str, Any] = {
            "schema_version": "0.1",
            "timestamp": datetime.utcnow().isoformat() + "Z",
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
                "n_mapped_removed": int(len(res.mapping)) if res.mapping is not None else 0,
            },
            "columns": _safe_jsonable(cols),
        }

        if getattr(res, "stats", None) is not None:
            report["reducer_stats"] = _safe_jsonable(res.stats)

        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
