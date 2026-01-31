from __future__ import annotations
import argparse

from biosieve.core import run_reduce
from biosieve.types import Columns

def add_reduce_parser(sub: argparse._SubParsersAction) -> None:
    pr = sub.add_parser("reduce", help="Reduce redundancy.")
    pr.add_argument("--in", dest="in_path", required=True)
    pr.add_argument("--out", dest="out_path", required=True)
    pr.add_argument("--map", dest="map_path", default=None)
    pr.add_argument("--report", dest="report_path", default=None)
    pr.add_argument("--strategy", required=True)
    pr.add_argument("--id-col", default="id")
    pr.add_argument("--seq-col", default="sequence")
    pr.add_argument("--seed", type=int, default=13)

    def _run(args, registry):
        cols = Columns(id_col=args.id_col, seq_col=args.seq_col)
        run_reduce(
            registry=registry,
            in_path=args.in_path,
            out_path=args.out_path,
            cols=cols,
            strategy=args.strategy,
            map_path=args.map_path,
            report_path=args.report_path,
            seed=args.seed,
        )

    pr.set_defaults(func=_run)
