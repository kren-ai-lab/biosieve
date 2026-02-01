from __future__ import annotations

import argparse
from typing import Any, Dict

from biosieve.core.split_runner import run_split
from biosieve.io.params import load_params, params_for_strategy
from biosieve.types import Columns
from biosieve.core.registry import StrategyRegistry


def add_split_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "split",
        help="Split a dataset into train/test(/val) using a selected strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--in", dest="in_path", required=True, help="Input CSV path.")
    p.add_argument("--outdir", required=True, help="Output directory for split CSVs.")
    p.add_argument("--strategy", required=True, help="Split strategy name (random, stratified).")
    p.add_argument("--report", dest="report_path", default=None, help="Optional JSON report path.")

    p.add_argument("--id-col", default="id", help="Column name for unique sample ids.")
    p.add_argument("--seq-col", default="sequence", help="Column name for sequences (if applicable).")

    p.add_argument("--params", dest="params_path", default=None, help="YAML/JSON file with strategy parameters.")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="Override params: --set random.seed=13")

    p.add_argument("--sep", default=",", help="CSV delimiter used to read input.")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding used to read input.")

    p.set_defaults(func=_run_split)


def _run_split(args: argparse.Namespace, registry: StrategyRegistry) -> None:
    cols = Columns(id_col=args.id_col, seq_col=args.seq_col)

    all_params = load_params(args.params_path, overrides=args.overrides)
    strat_params: Dict[str, Any] = params_for_strategy(all_params, args.strategy)

    read_csv_kwargs = {"sep": args.sep, "encoding": args.encoding}

    run_split(
        in_path=args.in_path,
        outdir=args.outdir,
        strategy=args.strategy,
        registry=registry,
        cols=cols,
        report_path=args.report_path,
        strategy_params=strat_params,
        read_csv_kwargs=read_csv_kwargs,
    )
