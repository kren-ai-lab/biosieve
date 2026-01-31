from __future__ import annotations
import argparse

from biosieve.core import build_registry
from biosieve.cli.reduce import add_reduce_parser
from biosieve.cli.split import add_split_parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="biosieve", description="BioSieve CLI")
    sub = p.add_subparsers(dest="command", required=True)
    add_reduce_parser(sub)
    add_split_parser(sub)
    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args, build_registry())
