from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from biosieve.core.factory import instantiate_strategy
from biosieve.core.registry import StrategyRegistry
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

log = get_logger(__name__)


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp with a trailing Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: str) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to CSV (UTF-8, no index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a dict to JSON (UTF-8, pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_input_df(df: pd.DataFrame, cols: Columns) -> None:
    """
    Validate required columns and id uniqueness.

    Raises
    ------
    ValueError
        If id column is missing or ids are not unique.
    """
    if cols.id_col not in df.columns:
        raise ValueError(f"Missing id column '{cols.id_col}' in input data. Columns: {df.columns.tolist()}")

    n_in = len(df)
    unique_ids = df[cols.id_col].astype(str).nunique()
    if unique_ids != n_in:
        raise ValueError(
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{cols.id_col}'."
        )


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
    """
    Run a splitting strategy and export split artefacts to disk.

    Artefact contract
    -----------------
    Single split (strategies implementing `run(df, cols)`):
      - outdir/train.csv
      - outdir/test.csv
      - outdir/val.csv (optional)
      - outdir/split_report.json (default)

    K-fold split (strategies implementing `run_folds(df, cols)`):
      - outdir/fold_00/train.csv, test.csv, val.csv (optional)
      - outdir/fold_01/...
      - ...
      - outdir/kfold_report.json (default)

    Parameters
    ----------
    in_path:
        Path to input CSV file.
    outdir:
        Output directory where split files and reports are written.
    strategy:
        Split strategy name (must exist in `registry.splitters`).
    registry:
        Strategy registry holding available splitters (classes/specs).
    cols:
        Columns spec. If None, defaults to Columns(id_col="id", seq_col="sequence").
    report_path:
        Optional custom path for report JSON.
        - For single split: defaults to outdir/split_report.json
        - For k-fold: defaults to outdir/kfold_report.json
    strategy_params:
        Parameters to instantiate the strategy dataclass.
        Unknown keys raise ValueError (strict contract).
    read_csv_kwargs:
        Extra kwargs passed to pandas.read_csv (e.g., sep, dtype, usecols).

    Raises
    ------
    ValueError
        If strategy is unknown, required columns are missing, ids are not unique,
        or the splitter returns invalid outputs.
    ImportError
        If a strategy requires optional dependencies (e.g., scikit-learn) that are missing.
    FileNotFoundError
        If `in_path` does not exist (raised by pandas).
    """
    t0 = time.time()

    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    read_csv_kwargs = read_csv_kwargs or {}

    # Validate strategy name early (avoid silent typos)
    if not registry.has_splitter(strategy):
        available = sorted(list(registry.list_splitters().keys()))
        raise ValueError(f"Unknown split strategy '{strategy}'. Available: {available}")

    out = _ensure_dir(outdir)

    log.info("split:start | strategy=%s | in=%s | outdir=%s", strategy, in_path, str(out))
    log.info("split:params | %s", strategy_params)

    # Read + validate input
    df = pd.read_csv(in_path, **read_csv_kwargs)
    _validate_input_df(df, cols)

    log.info("split:input | n_rows=%d | n_cols=%d", len(df), len(df.columns))
    log.debug("split:columns | %s", df.columns.tolist())

    # Instantiate strategy (lazy-safe)
    splitter_cls = registry.get_splitter_class(strategy)
    log.debug("split:strategy_class | %s.%s", splitter_cls.__module__, splitter_cls.__name__)
    splitter = instantiate_strategy(splitter_cls, strategy_params)

    # ----------------------------
    # K-fold mode: splitter.run_folds
    # ----------------------------
    if hasattr(splitter, "run_folds") and callable(getattr(splitter, "run_folds")):
        log.info("split:mode | kfold")
        folds = splitter.run_folds(df, cols)  # type: ignore[attr-defined]

        if not isinstance(folds, list) or len(folds) == 0:
            raise ValueError(
                f"Splitter '{strategy}' returned an invalid folds object. "
                "Expected a non-empty list of SplitResult."
            )

        folds_meta: List[Dict[str, Any]] = []
        for i, res in enumerate(folds):
            fold_idx = int(res.stats.get("fold_index", i)) if isinstance(res.stats, dict) else i

            fdir = _ensure_dir(str(out / f"fold_{fold_idx:02d}"))
            _write_csv(fdir / "train.csv", res.train)
            _write_csv(fdir / "test.csv", res.test)
            if res.val is not None:
                _write_csv(fdir / "val.csv", res.val)

            n_train = len(res.train)
            n_test = len(res.test)
            n_val = len(res.val) if res.val is not None else 0
            log.info("split:fold | idx=%d | train=%d | val=%d | test=%d", fold_idx, n_train, n_val, n_test)

            folds_meta.append(
                {
                    "fold_index": fold_idx,
                    "split_params": res.params,
                    "stats": res.stats,
                }
            )

        rp = Path(report_path) if report_path else (out / "kfold_report.json")
        report = {
            "schema_version": "0.1",
            "timestamp": _utc_timestamp(),
            "in_path": str(in_path),
            "outdir": str(out),
            "strategy": strategy,
            "strategy_params": strategy_params,
            "kfold": True,
            "n_folds": len(folds_meta),
            "folds": folds_meta,
            "columns": {
                "id_col": cols.id_col,
                "seq_col": cols.seq_col,
                "label_col": cols.label_col,
                "group_col": cols.group_col,
                "cluster_col": cols.cluster_col,
                "date_col": cols.date_col,
            },
        }
        _write_json(rp, report)

        elapsed = time.time() - t0
        log.info("split:end | mode=kfold | seconds=%.3f | report=%s", elapsed, str(rp))
        return

    # ----------------------------
    # Single split mode: splitter.run
    # ----------------------------
    log.info("split:mode | single")
    res = splitter.run(df, cols)

    _write_csv(out / "train.csv", res.train)
    _write_csv(out / "test.csv", res.test)
    if res.val is not None:
        _write_csv(out / "val.csv", res.val)

    n_train = len(res.train)
    n_test = len(res.test)
    n_val = len(res.val) if res.val is not None else 0
    log.info("split:result | train=%d | val=%d | test=%d", n_train, n_val, n_test)

    rp = Path(report_path) if report_path else (out / "split_report.json")
    report = {
        "schema_version": "0.1",
        "timestamp": _utc_timestamp(),
        "in_path": str(in_path),
        "outdir": str(out),
        "strategy": strategy,
        "strategy_params": strategy_params,
        "kfold": False,
        "split_params": res.params,
        "stats": res.stats,
        "columns": {
            "id_col": cols.id_col,
            "seq_col": cols.seq_col,
            "label_col": cols.label_col,
            "group_col": cols.group_col,
            "cluster_col": cols.cluster_col,
            "date_col": cols.date_col,
        },
    }
    _write_json(rp, report)

    elapsed = time.time() - t0
    log.info("split:end | mode=single | seconds=%.3f | report=%s", elapsed, str(rp))
