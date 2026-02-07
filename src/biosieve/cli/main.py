from __future__ import annotations

import argparse
import sys
from typing import Optional

from biosieve.core.strategies import build_registry, build_registry_light

from biosieve.cli.reduce import add_reduce_subcommand
from biosieve.cli.split import add_split_subcommand
from biosieve.cli.info import add_info_subcommand


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

    add_reduce_subcommand(subparsers)
    add_split_subcommand(subparsers)
    add_info_subcommand(subparsers)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 2

    # Choose light registry for commands that must not import heavy deps.
    cmd = getattr(args, "command", None)
    registry = build_registry_light() if cmd in {"info", "validate"} else build_registry()

    try:
        args.func(args, registry)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130  # standard for SIGINT
    except Exception as e:
        # Later we'll route this through logger + --log-level/--quiet
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
