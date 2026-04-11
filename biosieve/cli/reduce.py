from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from biosieve.core.runner import run_reduce
from biosieve.io.params import load_params, params_for_strategy
from biosieve.types import Columns
from biosieve.core.registry import StrategyRegistry  # ajusta si tu path es distinto


def add_reduce_subcommand(subparsers: argparse._SubParsersAction) -> None:
    """
    Register `biosieve reduce` subcommand.
    """
    p = subparsers.add_parser(
        "reduce",
        help="Reduce redundancy in a dataset using a selected strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--in", dest="in_path", required=True, help="Input CSV path.")
    p.add_argument("--out", dest="out_path", required=True, help="Output CSV path (non-redundant).")
    p.add_argument("--strategy", required=True, help="Reducer strategy name (e.g., exact, mmseqs2, embedding_cosine).")

    # Optional outputs
    p.add_argument("--map", dest="map_path", default=None, help="CSV mapping path (removed_id -> representative_id).")
    p.add_argument("--report", dest="report_path", default=None, help="JSON report path.")

    # Column config
    p.add_argument("--id-col", default="id", help="Column name for unique sample ids.")
    p.add_argument("--seq-col", default="sequence", help="Column name for sequences (if applicable).")

    # Params system
    p.add_argument(
        "--params",
        dest="params_path",
        default=None,
        help="YAML/JSON file with strategy parameters. Format: {strategy_name: {param: value}}",
    )
    p.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override params. Example: --set embedding_cosine.threshold=0.97",
    )

    # CSV reading kwargs (lightweight)
    p.add_argument("--sep", default=",", help="CSV delimiter used to read input.")
    p.add_argument("--encoding", default="utf-8", help="CSV encoding used to read input.")

    p.set_defaults(func=_run_reduce)


def _run_reduce(args: argparse.Namespace, registry: StrategyRegistry) -> None:
    """
    Handler executed by main CLI.
    """
    cols = Columns(id_col=args.id_col, seq_col=args.seq_col)

    # Load and resolve params
    all_params = load_params(args.params_path, overrides=args.overrides)
    strat_params: Dict[str, Any] = params_for_strategy(all_params, args.strategy)

    # Read CSV kwargs
    read_csv_kwargs = {
        "sep": args.sep,
        "encoding": args.encoding,
    }

    run_reduce(
        in_path=args.in_path,
        out_path=args.out_path,
        strategy=args.strategy,
        registry=registry,
        cols=cols,
        map_path=args.map_path,
        report_path=args.report_path,
        strategy_params=strat_params,
        read_csv_kwargs=read_csv_kwargs,
    )
