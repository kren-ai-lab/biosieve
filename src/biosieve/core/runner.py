from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from biosieve.core.factory import instantiate_strategy
from biosieve.types import Columns
from biosieve.core.registry import StrategyRegistry

def _ensure_parent(path: Optional[str]) -> None:
    if not path:
        return
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)


def _safe_jsonable(x: Any) -> Any:
    """
    Make objects JSON-serializable (best-effort) for reports.
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
    # fallback
    return str(x)


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
    Execute redundancy reduction.

    Parameters
    ----------
    in_path:
        Input CSV path.
    out_path:
        Output CSV path for non-redundant dataset.
    strategy:
        Strategy name (e.g., 'mmseqs2', 'embedding_cosine', ...).
    registry:
        StrategyRegistry containing reducer classes: registry.reducers[strategy] -> class
    cols:
        Columns specification. If None, defaults to Columns(id_col="id", seq_col="sequence").
    map_path:
        Optional mapping CSV path (removed_id -> representative_id + metadata).
    report_path:
        Optional JSON report path.
    strategy_params:
        Dict of params used to instantiate the reducer class.
    read_csv_kwargs:
        Optional kwargs passed to pandas.read_csv.
    """
    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    read_csv_kwargs = read_csv_kwargs or {}

    # Validate strategy exists
    if strategy not in registry.reducers:
        available = sorted(list(registry.reducers.keys()))
        raise ValueError(f"Unknown reducer strategy '{strategy}'. Available: {available}")

    _ensure_parent(out_path)
    _ensure_parent(map_path)
    _ensure_parent(report_path)

    # Load dataset
    df = pd.read_csv(in_path, **read_csv_kwargs)
    if cols.id_col not in df.columns:
        raise ValueError(f"Missing id column '{cols.id_col}' in input data. Columns: {df.columns.tolist()}")
    if cols.seq_col not in df.columns:
        # Some reducers may not require sequences (e.g., descriptor/structural). Still keep as soft check.
        # We allow missing sequence column if reducer doesn't use it, but keep warning-like behavior by not raising here.
        pass

    n_in = len(df)
    unique_ids = df[cols.id_col].astype(str).nunique()
    if unique_ids != n_in:
        raise ValueError(
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{cols.id_col}'."
        )

    # Instantiate reducer from class using params
    reducer_cls = registry.reducers[strategy]
    reducer = instantiate_strategy(reducer_cls, strategy_params)

    # Run reduction
    res = reducer.run(df, cols)  # ReductionResult

    # Write outputs
    res.df.to_csv(out_path, index=False)

    if map_path is not None:
        if res.mapping is None:
            # keep consistent: write empty mapping if reducer didn't produce it
            pd.DataFrame(columns=["removed_id", "representative_id", "cluster_id", "score"]).to_csv(map_path, index=False)
        else:
            res.mapping.to_csv(map_path, index=False)

    # Report
    if report_path is not None:
        n_out = len(res.df)
        n_removed = n_in - n_out

        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "in_path": str(in_path),
            "out_path": str(out_path),
            "map_path": str(map_path) if map_path else None,
            "strategy": strategy,
            "strategy_params": _safe_jsonable(strategy_params),
            "reducer_params": _safe_jsonable(getattr(res, "params", {})),
            "stats": {
                "n_in": int(n_in),
                "n_out": int(n_out),
                "n_removed": int(n_removed),
                "fraction_removed": float(n_removed / n_in) if n_in else 0.0,
            },
            "columns": _safe_jsonable(cols),
        }

        # Optional: mapping stats
        if getattr(res, "mapping", None) is not None:
            report["stats"]["n_mapped_removed"] = int(len(res.mapping))
        else:
            report["stats"]["n_mapped_removed"] = 0

        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
