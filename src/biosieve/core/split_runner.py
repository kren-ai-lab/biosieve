from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from biosieve.core.factory import instantiate_strategy
from biosieve.core.registry import StrategyRegistry
from biosieve.types import Columns


def _ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def run_split(
    in_path: str,
    outdir: str,
    strategy: str,
    registry: StrategyRegistry,
    *,
    cols: Optional[Columns] = None,
    report_path: Optional[str] = None,
    strategy_params: Optional[Dict[str, Any]] = None,
    read_csv_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    read_csv_kwargs = read_csv_kwargs or {}

    if strategy not in registry.splitters:
        available = sorted(list(registry.splitters.keys()))
        raise ValueError(f"Unknown split strategy '{strategy}'. Available: {available}")

    out = _ensure_dir(outdir)

    df = pd.read_csv(in_path, **read_csv_kwargs)
    if cols.id_col not in df.columns:
        raise ValueError(f"Missing id column '{cols.id_col}' in input data. Columns: {df.columns.tolist()}")

    n_in = len(df)
    unique_ids = df[cols.id_col].astype(str).nunique()
    if unique_ids != n_in:
        raise ValueError(
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{cols.id_col}'."
        )

    splitter_cls = registry.splitters[strategy]
    splitter = instantiate_strategy(splitter_cls, strategy_params)

    res = splitter.run(df, cols)

    (out / "train.csv").write_text(res.train.to_csv(index=False), encoding="utf-8")
    (out / "test.csv").write_text(res.test.to_csv(index=False), encoding="utf-8")
    if res.val is not None:
        (out / "val.csv").write_text(res.val.to_csv(index=False), encoding="utf-8")

    # report path default
    rp = Path(report_path) if report_path else (out / "split_report.json")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "in_path": str(in_path),
        "outdir": str(out),
        "strategy": strategy,
        "strategy_params": strategy_params,
        "split_params": res.params,
        "stats": res.stats,
        "columns": {"id_col": cols.id_col, "seq_col": cols.seq_col},
    }
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
