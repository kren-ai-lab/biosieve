from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from biosieve.reduction.base import Reducer
from biosieve.splitting.base import Splitter

@dataclass(frozen=True)
class StrategyRegistry:
    reducers: Dict[str, Reducer]
    splitters: Dict[str, Splitter]

    def get_reducer(self, name: str) -> Reducer:
        if name not in self.reducers:
            raise ValueError(f"Unknown reduction strategy '{name}'. Available: {sorted(self.reducers)}")
        return self.reducers[name]

    def get_splitter(self, name: str) -> Splitter:
        if name not in self.splitters:
            raise ValueError(f"Unknown split strategy '{name}'. Available: {sorted(self.splitters)}")
        return self.splitters[name]
