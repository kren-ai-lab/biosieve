from __future__ import annotations

from biosieve.core.registry import StrategyRegistry
from biosieve.splitting import RandomSplit, StratifiedSplit, SplitFractions
from biosieve.reduction import (ExactDedupReducer, IdentityGreedyReducer, KmerJaccardReducer,
                                MMseqs2Reducer, EmbeddingCosineReducer)

def build_registry() -> StrategyRegistry:
    reducers = {
        "exact": ExactDedupReducer(),
        "identity_greedy": IdentityGreedyReducer(threshold=0.9, k=5, jaccard_prefilter=0.2),
        "kmer_jaccard": KmerJaccardReducer(threshold=0.7, k=5),
        "mmseqs2": MMseqs2Reducer(min_seq_id=0.9, coverage=0.8, threads=8, keep_tmp=False),
        "embedding_cosine": EmbeddingCosineReducer(
            embeddings_path="embeddings.npy",
            ids_path="embedding_ids.csv",
            ids_col="id",
            threshold=0.95,
            use_faiss=True,
            n_jobs=8,
            dtype="float32",
        ),
    }
    splitters = {
        "random": RandomSplit(SplitFractions()),
        "stratified": StratifiedSplit(SplitFractions(), label_col="label"),
    }
    return StrategyRegistry(reducers=reducers, splitters=splitters)
