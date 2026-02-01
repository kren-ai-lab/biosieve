from __future__ import annotations

from biosieve.core.registry import StrategyRegistry
from biosieve.splitting import RandomSplit, StratifiedSplit, SplitFractions
from biosieve.reduction import (ExactDedupReducer, IdentityGreedyReducer, KmerJaccardReducer,
                                MMseqs2Reducer, EmbeddingCosineReducer, DescriptorEuclideanReducer,
                                StructuralDistanceReducer)

def build_registry() -> StrategyRegistry:
    reducers = {
        "exact": ExactDedupReducer,
        "identity_greedy": IdentityGreedyReducer,
        "kmer_jaccard": KmerJaccardReducer,
        "mmseqs2": MMseqs2Reducer,
        "embedding_cosine": EmbeddingCosineReducer,
        "descriptor_euclidean": DescriptorEuclideanReducer,
        "structural_distance": StructuralDistanceReducer,
    }
    splitters = {
        "random": RandomSplit(SplitFractions()),
        "stratified": StratifiedSplit(SplitFractions(), label_col="label"),
    }
    return StrategyRegistry(reducers=reducers, splitters=splitters)
