from __future__ import annotations

from biosieve.core.registry import StrategyRegistry
from biosieve.splitting import RandomSplit, StratifiedSplit, SplitFractions
from biosieve.reduction import (ExactDedupReducer, IdentityGreedyReducer, KmerJaccardReducer,
                                MMseqs2Reducer)

def build_registry() -> StrategyRegistry:
    reducers = {
        "exact": ExactDedupReducer(),
        "identity_greedy": IdentityGreedyReducer(threshold=0.9, k=5, jaccard_prefilter=0.2),
        "kmer_jaccard": KmerJaccardReducer(threshold=0.7, k=5),
        "mmseqs2": MMseqs2Reducer(min_seq_id=0.9, coverage=0.8, threads=8, keep_tmp=False),
    }
    splitters = {
        "random": RandomSplit(SplitFractions()),
        "stratified": StratifiedSplit(SplitFractions(), label_col="label"),
    }
    return StrategyRegistry(reducers=reducers, splitters=splitters)
