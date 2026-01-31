from __future__ import annotations
import argparse

from biosieve.core import run_split
from biosieve.types import Columns

def add_split_parser(sub: argparse._SubParsersAction) -> None:
    ps = sub.add_parser("split", help="Create train/val/test splits.")
    ps.add_argument("--in", dest="in_path", required=True)
    ps.add_argument("--outdir", required=True)
    ps.add_argument("--strategy", required=True)
    ps.add_argument("--seed", type=int, default=13)
    ps.add_argument("--id-col", default="id")
    ps.add_argument("--seq-col", default="sequence")
    ps.add_argument("--label-col", default="label")
    ps.add_argument("--group-col", default="group")
    ps.add_argument("--cluster-col", default="cluster_id")
    ps.add_argument("--date-col", default="date")
    ps.add_argument("--report", dest="report_path", default=None)

    def _run(args, registry):
        cols = Columns(
            id_col=args.id_col,
            seq_col=args.seq_col,
            label_col=args.label_col,
            group_col=args.group_col,
            cluster_col=args.cluster_col,
            date_col=args.date_col,
        )
        run_split(
            registry=registry,
            in_path=args.in_path,
            outdir=args.outdir,
            cols=cols,
            strategy=args.strategy,
            seed=args.seed,
            report_path=args.report_path,
        )

    ps.set_defaults(func=_run)
