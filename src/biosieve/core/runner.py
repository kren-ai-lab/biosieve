from __future__ import annotations
import os
from typing import Optional

from biosieve.core.registry import StrategyRegistry
from biosieve.io import read_csv, write_csv, validate_required_columns, normalize_sequences
from biosieve.reporting import ReductionReport, SplitReport, write_json, write_assignments
from biosieve.types import Columns
from biosieve.utils.logging import setup_logger

logger = setup_logger()

def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def run_reduce(
    registry: StrategyRegistry,
    in_path: str,
    out_path: str,
    cols: Columns,
    strategy: str,
    map_path: Optional[str] = None,
    report_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> None:
    df = read_csv(in_path)
    validate_required_columns(df, cols)
    df = normalize_sequences(df, cols.seq_col)

    reducer = registry.get_reducer(strategy)
    res = reducer.run(df, cols)

    write_csv(res.df, out_path)
    if map_path:
        write_csv(res.mapping, map_path)

    if report_path:
        rep = ReductionReport(
            strategy=res.strategy,
            n_input=len(df),
            n_output=len(res.df),
            n_removed=len(res.mapping),
            params=res.params,
            seed=seed,
        )
        write_json(rep, report_path)

    logger.info(f"reduce[{strategy}] {len(df)} -> {len(res.df)}")

def run_split(
    registry: StrategyRegistry,
    in_path: str,
    outdir: str,
    cols: Columns,
    strategy: str,
    seed: int,
    report_path: Optional[str] = None,
) -> None:
    df = read_csv(in_path)
    validate_required_columns(df, cols)
    df = normalize_sequences(df, cols.seq_col)

    splitter = registry.get_splitter(strategy)
    res = splitter.run(df, cols, seed=seed)

    _ensure_outdir(outdir)
    write_csv(res.train, os.path.join(outdir, "train.csv"))
    write_csv(res.val, os.path.join(outdir, "val.csv"))
    write_csv(res.test, os.path.join(outdir, "test.csv"))
    write_assignments(res.assignments, os.path.join(outdir, "split_assignments.csv"))

    rp = report_path or os.path.join(outdir, "split_report.json")
    rep = SplitReport(
        strategy=res.strategy,
        n_total=len(df),
        n_train=len(res.train),
        n_val=len(res.val),
        n_test=len(res.test),
        params=res.params,
        seed=res.seed,
    )
    write_json(rep, rp)

    logger.info(f"split[{strategy}] train={len(res.train)} val={len(res.val)} test={len(res.test)}")
