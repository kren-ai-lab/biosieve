from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, Union

from biosieve.core.spec import StrategySpec, lazy_import_class

ReducerLike = Union[Type[Any], StrategySpec]
SplitterLike = Union[Type[Any], StrategySpec]


@dataclass
class StrategyRegistry:
    """
    Registry of available reducer and splitter strategies.

    Supports both:
    - eager class registration (Type)
    - lazy registration via StrategySpec (import path)

    Notes
    -----
    - `list_*` is safe for light registries (does not import classes).
    - `get_*` may import classes for StrategySpec entries.
    """

    reducers: Dict[str, ReducerLike] = field(default_factory=dict)
    splitters: Dict[str, SplitterLike] = field(default_factory=dict)

    def add_reducer(self, name: str, reducer: ReducerLike) -> None:
        self.reducers[name] = reducer

    def add_splitter(self, name: str, splitter: SplitterLike) -> None:
        self.splitters[name] = splitter

    def has_reducer(self, name: str) -> bool:
        return name in self.reducers

    def has_splitter(self, name: str) -> bool:
        return name in self.splitters

    def list_reducers(self) -> Dict[str, ReducerLike]:
        return dict(self.reducers)

    def list_splitters(self) -> Dict[str, SplitterLike]:
        return dict(self.splitters)

    def get_reducer_class(self, name: str) -> Type[Any]:
        if name not in self.reducers:
            raise KeyError(f"Unknown reducer strategy '{name}'. Available: {sorted(self.reducers)}")
        obj = self.reducers[name]
        if isinstance(obj, StrategySpec):
            cls = lazy_import_class(obj.import_path)
            # cache resolved class for future calls (safe)
            self.reducers[name] = cls
            return cls
        return obj

    def get_splitter_class(self, name: str) -> Type[Any]:
        if name not in self.splitters:
            raise KeyError(f"Unknown splitter strategy '{name}'. Available: {sorted(self.splitters)}")
        obj = self.splitters[name]
        if isinstance(obj, StrategySpec):
            cls = lazy_import_class(obj.import_path)
            self.splitters[name] = cls
            return cls
        return obj

    def get_spec(self, name: str, kind: str) -> Optional[StrategySpec]:
        """
        Return StrategySpec if the entry is lazy, else None.
        """
        if kind == "reducer":
            obj = self.reducers.get(name)
        elif kind == "splitter":
            obj = self.splitters.get(name)
        else:
            raise ValueError("kind must be 'reducer' or 'splitter'")

        return obj if isinstance(obj, StrategySpec) else None
