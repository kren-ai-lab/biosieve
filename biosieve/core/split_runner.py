"""Execution runner for split workflows."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import polars as pl

from biosieve.core.factory import instantiate_strategy
from biosieve.splitting.base import KFoldSplitter, SplitResult, Splitter
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.core.registry import StrategyRegistry

log = get_logger(__name__)


def _utc_timestamp() -> str:
    """Return an ISO 8601 UTC timestamp with a trailing Z."""
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _ensure_dir(path: str) -> Path:
    """Ensure a directory exists and return its Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_csv(path: Path, df: pl.DataFrame) -> None:
    """Write a DataFrame to CSV (UTF-8, no index)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a dict to JSON (UTF-8, pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_input_df(df: pl.DataFrame, cols: Columns) -> None:
    """Validate required columns and id uniqueness.

    Raises:
        ValueError: If id column is missing or ids are not unique.

    """
    if cols.id_col not in df.columns:
        msg = f"Missing id column '{cols.id_col}' in input data. Columns: {df.columns}"
        raise ValueError(msg)

    n_in = df.height
    unique_ids = df[cols.id_col].cast(pl.String).n_unique()
    if unique_ids != n_in:
        msg = (
            f"Input ids are not unique: {unique_ids} unique ids for {n_in} rows. "
            f"BioSieve expects unique '{cols.id_col}'."
        )
        raise ValueError(msg)


def _columns_payload(cols: Columns) -> dict[str, str | None]:
    return {
        "id_col": cols.id_col,
        "seq_col": cols.seq_col,
        "label_col": cols.label_col,
        "group_col": cols.group_col,
        "cluster_col": cols.cluster_col,
        "date_col": cols.date_col,
    }


def _write_split_outputs(out: Path, res: SplitResult) -> None:
    _write_csv(out / "train.csv", res.train)
    _write_csv(out / "test.csv", res.test)
    if res.val is not None:
        _write_csv(out / "val.csv", res.val)


def _build_kfold_report(
    *,
    in_path: str,
    out: Path,
    strategy: str,
    strategy_params: dict[str, object],
    cols: Columns,
    folds_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "timestamp": _utc_timestamp(),
        "in_path": str(in_path),
        "outdir": str(out),
        "strategy": strategy,
        "strategy_params": strategy_params,
        "kfold": True,
        "n_folds": len(folds_meta),
        "folds": folds_meta,
        "columns": _columns_payload(cols),
    }


def _build_single_report(
    *,
    in_path: str,
    out: Path,
    strategy: str,
    strategy_params: dict[str, object],
    cols: Columns,
    res: SplitResult,
) -> dict[str, Any]:
    return {
        "schema_version": "0.1",
        "timestamp": _utc_timestamp(),
        "in_path": str(in_path),
        "outdir": str(out),
        "strategy": strategy,
        "strategy_params": strategy_params,
        "kfold": False,
        "split_params": res.params,
        "stats": res.stats,
        "columns": _columns_payload(cols),
    }


def _run_kfold(
    *,
    splitter: KFoldSplitter,
    df: pl.DataFrame,
    cols: Columns,
    strategy: str,
    out: Path,
    report_path: str | None,
    in_path: str,
    strategy_params: dict[str, object],
) -> Path:
    folds = splitter.run_folds(df, cols)
    if not isinstance(folds, list) or len(folds) == 0:
        msg = (
            f"Splitter '{strategy}' returned an invalid folds object. "
            "Expected a non-empty list of SplitResult."
        )
        raise ValueError(msg)

    folds_meta: list[dict[str, Any]] = []
    for i, res in enumerate(folds):
        fold_idx = int(res.stats.get("fold_index", i)) if isinstance(res.stats, dict) else i
        fdir = _ensure_dir(str(out / f"fold_{fold_idx:02d}"))
        _write_split_outputs(fdir, res)
        n_val = res.val.height if res.val is not None else 0
        log.info(
            "split:fold | idx=%d | train=%d | val=%d | test=%d",
            fold_idx,
            res.train.height,
            n_val,
            res.test.height,
        )
        folds_meta.append({"fold_index": fold_idx, "split_params": res.params, "stats": res.stats})

    report = _build_kfold_report(
        in_path=in_path,
        out=out,
        strategy=strategy,
        strategy_params=strategy_params,
        cols=cols,
        folds_meta=folds_meta,
    )
    rp = Path(report_path) if report_path else (out / "kfold_report.json")
    _write_json(rp, report)
    return rp


def _run_single(
    *,
    splitter: Splitter,
    df: pl.DataFrame,
    cols: Columns,
    strategy: str,
    out: Path,
    report_path: str | None,
    in_path: str,
    strategy_params: dict[str, object],
) -> Path:
    res = splitter.run(df, cols)
    _write_split_outputs(out, res)
    n_val = res.val.height if res.val is not None else 0
    log.info("split:result | train=%d | val=%d | test=%d", res.train.height, n_val, res.test.height)
    report = _build_single_report(
        in_path=in_path,
        out=out,
        strategy=strategy,
        strategy_params=strategy_params,
        cols=cols,
        res=res,
    )
    rp = Path(report_path) if report_path else (out / "split_report.json")
    _write_json(rp, report)
    return rp


def run_split(
    in_path: str,
    outdir: str,
    strategy: str,
    registry: StrategyRegistry,
    *,
    cols: Columns | None = None,
    report_path: str | None = None,
    strategy_params: dict[str, object] | None = None,
    read_csv_kwargs: dict[str, object] | None = None,
) -> None:
    """Run a splitting strategy and export split artefacts to disk.

    Artefact contract:
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

    Args:
        in_path: Path to input CSV file.
        outdir: Output directory where split files and reports are written.
        strategy: Split strategy name (must exist in `registry.splitters`).
        registry: Strategy registry holding available splitters (classes/specs).
        cols: Columns spec. If None, defaults to Columns(id_col="id", seq_col="sequence").
        report_path:
            Optional custom path for report JSON.
            - For single split: defaults to outdir/split_report.json
            - For k-fold: defaults to outdir/kfold_report.json
        strategy_params:
            Parameters to instantiate the strategy dataclass.
            Unknown keys raise ValueError (strict contract).
        read_csv_kwargs: Extra kwargs passed to polars.read_csv (e.g., separator, dtypes, columns).

    Raises:
        ValueError: If strategy is unknown, required columns are missing, ids are not unique,
        or the splitter returns invalid outputs.
        ImportError: If a strategy requires optional dependencies (e.g., scikit-learn) that are missing.
        FileNotFoundError: If `in_path` does not exist (raised by polars).

    """
    t0 = time.time()

    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    read_csv_kwargs = read_csv_kwargs or {}
    if "sep" in read_csv_kwargs and "separator" not in read_csv_kwargs:
        read_csv_kwargs = {**read_csv_kwargs, "separator": read_csv_kwargs["sep"]}
        read_csv_kwargs.pop("sep")

    # Validate strategy name early (avoid silent typos)
    if not registry.has_splitter(strategy):
        available = sorted(registry.list_splitters().keys())
        msg = f"Unknown split strategy '{strategy}'. Available: {available}"
        raise ValueError(msg)

    out = _ensure_dir(outdir)

    log.info("split:start | strategy=%s | in=%s | outdir=%s", strategy, in_path, str(out))
    log.info("split:params | %s", strategy_params)

    # Read + validate input
    df = pl.read_csv(in_path, **cast("dict[str, Any]", read_csv_kwargs))
    _validate_input_df(df, cols)

    log.info("split:input | n_rows=%d | n_cols=%d", df.height, len(df.columns))
    log.debug("split:columns | %s", df.columns)

    # Instantiate strategy (lazy-safe)
    splitter_cls = registry.get_splitter_class(strategy)
    log.debug("split:strategy_class | %s.%s", splitter_cls.__module__, splitter_cls.__name__)
    splitter = instantiate_strategy(splitter_cls, strategy_params)

    rp: Path
    if isinstance(splitter, KFoldSplitter):
        log.info("split:mode | kfold")
        rp = _run_kfold(
            splitter=splitter,
            df=df,
            cols=cols,
            strategy=strategy,
            out=out,
            report_path=report_path,
            in_path=in_path,
            strategy_params=strategy_params,
        )
    else:
        log.info("split:mode | single")
        rp = _run_single(
            splitter=cast("Splitter", splitter),
            df=df,
            cols=cols,
            strategy=strategy,
            out=out,
            report_path=report_path,
            in_path=in_path,
            strategy_params=strategy_params,
        )

    elapsed = time.time() - t0
    mode = "kfold" if isinstance(splitter, KFoldSplitter) else "single"
    log.info("split:end | mode=%s | seconds=%.3f | report=%s", mode, elapsed, str(rp))
