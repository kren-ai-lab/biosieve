"""Execution runner for split workflows."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import biosieve
from biosieve._provenance import utc_now, write_provenance
from biosieve.core.common import (
    check_paths,
    effective_params,
    ensure_dir,
    provenance_inputs,
    read_table,
    utc_timestamp,
    validate_unique_ids,
    write_json,
    write_table,
)
from biosieve.splitting.base import KFoldSplitter, SplitResult, Splitter
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    import polars as pl

    from biosieve.core.registry import StrategyRegistry

log = get_logger(__name__)


def _columns_payload(cols: Columns) -> dict[str, str | None]:
    return asdict(cols)


def _write_split_outputs(out: Path, res: SplitResult, *, id_col: str, ext: str) -> None:
    write_table(res.train, out / f"train{ext}", id_col)
    write_table(res.test, out / f"test{ext}", id_col)
    if res.val is not None:
        write_table(res.val, out / f"val{ext}", id_col)


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
        "tool": {"name": "biosieve", "version": biosieve.__version__},
        "timestamp": utc_timestamp(),
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
        "tool": {"name": "biosieve", "version": biosieve.__version__},
        "timestamp": utc_timestamp(),
        "in_path": str(in_path),
        "outdir": str(out),
        "strategy": strategy,
        "strategy_params": strategy_params,
        "kfold": False,
        "effective_params": res.params,
        "summary": {
            "n_total": res.train.height + res.test.height + (res.val.height if res.val is not None else 0),
            "n_train": res.train.height,
            "n_test": res.test.height,
            "n_val": res.val.height if res.val is not None else 0,
        },
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
    ext: str,
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
        fdir = ensure_dir(str(out / f"fold_{fold_idx:02d}"))
        _write_split_outputs(fdir, res, id_col=cols.id_col, ext=ext)
        n_val = res.val.height if res.val is not None else 0
        log.info(
            "split:fold | idx=%d | train=%d | val=%d | test=%d",
            fold_idx,
            res.train.height,
            n_val,
            res.test.height,
        )
        folds_meta.append({"fold_index": fold_idx, "effective_params": res.params, "stats": res.stats})

    report = _build_kfold_report(
        in_path=in_path,
        out=out,
        strategy=strategy,
        strategy_params=strategy_params,
        cols=cols,
        folds_meta=folds_meta,
    )
    rp = Path(report_path) if report_path else (out / "kfold_report.json")
    write_json(rp, report)
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
    ext: str,
) -> Path:
    res = splitter.run(df, cols)
    _write_split_outputs(out, res, id_col=cols.id_col, ext=ext)
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
    write_json(rp, report)
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
    paths: dict[str, str] | None = None,
) -> None:
    """Run a splitting strategy and export split artefacts to disk.

    Artefact contract:
    Single split (strategies implementing `run(df, cols)`):
      - outdir/train.csv
      - outdir/test.csv
      - outdir/val.csv (optional)
      - outdir/split_report.json (default)
      - <outdir>.provenance.json (sibling of outdir)
      Files are Parquet instead of CSV when the input is Parquet; `id` is always the first column.

    K-fold split (strategies implementing `run_folds(df, cols)`):
      - outdir/fold_00/train.csv, test.csv, val.csv (optional)
      - outdir/fold_01/...
      - ...
      - outdir/kfold_report.json (default)

    Args:
        in_path: Path to input CSV or Parquet file.
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
        paths: Input files for the strategy, keyed by strategy field (e.g. `embeddings_path`).

    Raises:
        ValueError: If strategy is unknown, required columns are missing, ids are not unique,
        or the splitter returns invalid outputs.
        ImportError: If a strategy requires optional dependencies (e.g., scikit-learn) that are missing.
        FileNotFoundError: If `in_path` does not exist (raised by polars).

    """
    t0 = time.time()
    created_at = utc_now()

    if cols is None:
        cols = Columns(id_col="id", seq_col="sequence")

    strategy_params = strategy_params or {}
    paths = paths or {}
    ext = ".parquet" if Path(in_path).suffix.lower() == ".parquet" else ".csv"

    # Validate strategy name early (avoid silent typos)
    if not registry.has_splitter(strategy):
        available = sorted(registry.list_splitters().keys())
        msg = f"Unknown split strategy '{strategy}'. Available: {available}"
        raise ValueError(msg)

    out = ensure_dir(outdir)

    log.info("split:start | strategy=%s | in=%s | outdir=%s", strategy, in_path, str(out))
    log.info("split:params | %s", strategy_params)

    # Read + validate input
    check_paths(registry.get_splitter_class(strategy), paths)
    df = read_table(in_path, cols.id_col, read_csv_kwargs)
    validate_unique_ids(df, cols.id_col)

    log.info("split:input | n_rows=%d | n_cols=%d", df.height, len(df.columns))
    log.debug("split:columns | %s", df.columns)

    # Instantiate strategy (lazy-safe)
    splitter_cls = registry.get_splitter_class(strategy)
    log.debug("split:strategy_class | %s.%s", splitter_cls.__module__, splitter_cls.__name__)
    splitter = registry.create_splitter(strategy, {**strategy_params, **paths})

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
            ext=ext,
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
            ext=ext,
        )

    write_provenance(
        out,
        tool="biosieve",
        version=biosieve.__version__,
        command="split",
        created_at=created_at,
        parameters={
            "strategy": strategy,
            "params": effective_params(splitter),
            "columns": asdict(cols),
            "read_options": read_csv_kwargs or {},
        },
        inputs=provenance_inputs(in_path, paths),
        packages=("biosieve", "polars", "numpy"),
        extensions={"strategy": strategy, "kfold": isinstance(splitter, KFoldSplitter), "n_rows": df.height},
    )

    elapsed = time.time() - t0
    mode = "kfold" if isinstance(splitter, KFoldSplitter) else "single"
    log.info("split:end | mode=%s | seconds=%.3f | report=%s", mode, elapsed, str(rp))
