from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
from typing import Any, Dict

from biosieve.core.registry import StrategyRegistry
from biosieve.core.spec import StrategySpec


def add_info_subcommand(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "info",
        help="List available strategies and their default parameters (safe, lazy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--kind", choices=["all", "reduce", "split"], default="all", help="Filter by kind.")
    p.add_argument(
        "--show-defaults", action="store_true", help="Show dataclass defaults (may import classes)."
    )
    p.set_defaults(func=_run_info)


def _defaults_for_cls(cls: Any) -> Dict[str, Any]:
    if not is_dataclass(cls):
        return {}
    out: Dict[str, Any] = {}
    for f in fields(cls):
        # Skip fields that are not init params if needed; for your dataclasses it's fine.
        out[f.name] = f.default
    return out


def _print_block(title: str, items: Dict[str, Any]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for name in sorted(items.keys()):
        obj = items[name]
        if isinstance(obj, StrategySpec):
            print(f"- {name}  [{obj.import_path}]")
        else:
            print(f"- {name}  [{obj.__module__}:{obj.__name__}]")


def _run_info(args: argparse.Namespace, registry: StrategyRegistry) -> None:
    if args.kind in {"all", "reduce"}:
        _print_block("Reducers", registry.list_reducers())

    if args.kind in {"all", "split"}:
        _print_block("Splitters", registry.list_splitters())

    if args.show_defaults:
        # This may import classes for StrategySpec entries (still OK if environment supports deps)
        if args.kind in {"all", "reduce"}:
            print("\nReducer defaults")
            print("--------------")
            for name in sorted(registry.list_reducers().keys()):
                cls = registry.get_reducer_class(name)
                d = _defaults_for_cls(cls)
                print(f"\n{name}")
                for k, v in d.items():
                    print(f"  {k}: {v}")

        if args.kind in {"all", "split"}:
            print("\nSplitter defaults")
            print("----------------")
            for name in sorted(registry.list_splitters().keys()):
                cls = registry.get_splitter_class(name)
                d = _defaults_for_cls(cls)
                print(f"\n{name}")
                for k, v in d.items():
                    print(f"  {k}: {v}")
