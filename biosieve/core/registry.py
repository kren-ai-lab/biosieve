"""Strategy registry and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from biosieve.core.spec import StrategySpec, lazy_import_class

ReducerLike = type[Any] | StrategySpec
SplitterLike = type[Any] | StrategySpec


@dataclass
class StrategyRegistry:
    """Registry of available reducer and splitter strategies.

    Supports both:
    - eager class registration (Type)
    - lazy registration via StrategySpec (import path)

    Notes:
        - `list_*` is safe for light registries (does not import classes).
        - `get_*` may import classes for StrategySpec entries.

    """

    reducers: dict[str, ReducerLike] = field(default_factory=dict)
    splitters: dict[str, SplitterLike] = field(default_factory=dict)

    def add_reducer(self, name: str, reducer: ReducerLike) -> None:
        """Register a reducer strategy."""
        self.reducers[name] = reducer

    def add_splitter(self, name: str, splitter: SplitterLike) -> None:
        """Register a splitter strategy."""
        self.splitters[name] = splitter

    def has_reducer(self, name: str) -> bool:
        """Return whether a reducer strategy exists."""
        return name in self.reducers

    def has_splitter(self, name: str) -> bool:
        """Return whether a splitter strategy exists."""
        return name in self.splitters

    def list_reducers(self) -> dict[str, ReducerLike]:
        """Return a copy of registered reducers."""
        return dict(self.reducers)

    def list_splitters(self) -> dict[str, SplitterLike]:
        """Return a copy of registered splitters."""
        return dict(self.splitters)

    def get_reducer_class(self, name: str) -> type[Any]:
        """Resolve and return a reducer class by name."""
        if name not in self.reducers:
            msg = f"Unknown reducer strategy '{name}'. Available: {sorted(self.reducers)}"
            raise KeyError(msg)
        obj = self.reducers[name]
        if isinstance(obj, StrategySpec):
            cls = lazy_import_class(obj.import_path)
            # cache resolved class for future calls (safe)
            self.reducers[name] = cls
            return cls
        return obj

    def get_splitter_class(self, name: str) -> type[Any]:
        """Resolve and return a splitter class by name."""
        if name not in self.splitters:
            msg = f"Unknown splitter strategy '{name}'. Available: {sorted(self.splitters)}"
            raise KeyError(msg)
        obj = self.splitters[name]
        if isinstance(obj, StrategySpec):
            cls = lazy_import_class(obj.import_path)
            self.splitters[name] = cls
            return cls
        return obj

    def get_spec(self, name: str, kind: str) -> StrategySpec | None:
        """Return StrategySpec if the entry is lazy, else None."""
        if kind == "reducer":
            obj = self.reducers.get(name)
        elif kind == "splitter":
            obj = self.splitters.get(name)
        else:
            msg = "kind must be 'reducer' or 'splitter'"
            raise ValueError(msg)

        return obj if isinstance(obj, StrategySpec) else None
