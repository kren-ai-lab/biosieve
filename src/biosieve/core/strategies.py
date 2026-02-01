from __future__ import annotations

from biosieve.core.registry import StrategyRegistry
from biosieve.reduction import (ExactDedupReducer, IdentityGreedyReducer, KmerJaccardReducer,
                                MMseqs2Reducer, EmbeddingCosineReducer, DescriptorEuclideanReducer,
                                StructuralDistanceReducer)

from biosieve.splitting import RandomSplitter, StratifiedSplitter

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
        "random": RandomSplitter,
        "stratified": StratifiedSplitter,
    }
    
    return StrategyRegistry(reducers=reducers, splitters=splitters)
