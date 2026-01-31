from __future__ import annotations

from biosieve.core.registry import StrategyRegistry
from biosieve.reduction import ExactDedupReducer
from biosieve.splitting import RandomSplit, StratifiedSplit, SplitFractions

def build_registry() -> StrategyRegistry:
    reducers = {
        "exact": ExactDedupReducer(),
    }
    splitters = {
        "random": RandomSplit(SplitFractions()),
        "stratified": StratifiedSplit(SplitFractions(), label_col="label"),
    }
    return StrategyRegistry(reducers=reducers, splitters=splitters)
