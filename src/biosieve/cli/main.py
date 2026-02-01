from __future__ import annotations

import argparse
import sys
from typing import Optional

from biosieve.core.strategies import build_registry

# Subcommands
from biosieve.cli.reduce import add_reduce_subcommand
from biosieve.cli.split import add_split_subcommand

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="biosieve",
        description="BioSieve: dataset splitting and redundancy reduction toolkit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--version",
        action="version",
        version="biosieve 0.1.0",
        help="Show version and exit.",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="",
    )

    # Register subcommands
    add_reduce_subcommand(subparsers)
    add_split_subcommand(subparsers)
    
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # If no subcommand provided, show help and exit with code 2 (argparse convention)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 2

    registry = build_registry()
    args.func(args, registry)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
