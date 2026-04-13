from __future__ import annotations

from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from typing import Any, Dict

import typer

from biosieve.core.registry import StrategyRegistry
from biosieve.core.spec import StrategySpec
from biosieve.core.strategies import build_registry

KIND_OPTION = typer.Option(
    "all",
    "--kind",
    help="Filter by kind. One of: all, reduce, split.",
    show_default=True,
)
SHOW_DEFAULTS_OPTION = typer.Option(
    False,
    "--show-defaults/--no-show-defaults",
    help="Show dataclass defaults (may import classes).",
    show_default=True,
)


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


def info(
    kind: str = KIND_OPTION,
    show_defaults: bool = SHOW_DEFAULTS_OPTION,
) -> None:
    """List available strategies and their default parameters."""
    if kind not in {"all", "reduce", "split"}:
        raise typer.BadParameter("kind must be one of: all, reduce, split")

    args = SimpleNamespace(kind=kind, show_defaults=show_defaults)
    _run_info(args, build_registry())


def _run_info(args: Any, registry: StrategyRegistry) -> None:
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
