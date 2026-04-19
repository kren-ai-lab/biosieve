"""Default strategy registry construction."""

from __future__ import annotations

from biosieve.core.registry import StrategyRegistry

REDUCERS: dict[str, str] = {
    "exact": "biosieve.reduction.exact:ExactDedupReducer",
    "kmer_jaccard": "biosieve.reduction.kmer_jaccard:KmerJaccardReducer",
    "minhash_jaccard": "biosieve.reduction.minhash_jaccard:MinHashJaccardReducer",
    "identity_greedy": "biosieve.reduction.identity_greedy:IdentityGreedyReducer",
    "mmseqs2": "biosieve.reduction.mmseqs2:MMseqs2Reducer",
    "embedding_cosine": "biosieve.reduction.embedding_cosine:EmbeddingCosineReducer",
    "descriptor_euclidean": "biosieve.reduction.descriptor_euclidean:DescriptorEuclideanReducer",
    "structural_distance": "biosieve.reduction.structural_distance:StructuralDistanceReducer",
}

SPLITTERS: dict[str, str] = {
    "random": "biosieve.splitting.random:RandomSplitter",
    "stratified": "biosieve.splitting.stratified:StratifiedSplitter",
    "stratified_numeric": "biosieve.splitting.stratified_numeric:StratifiedNumericSplitter",
    "group": "biosieve.splitting.group:GroupSplitter",
    "time": "biosieve.splitting.time_based:TimeSplitter",
    "distance_aware": "biosieve.splitting.distance_aware:DistanceAwareSplitter",
    "cluster_aware": "biosieve.splitting.cluster:ClusterAwareSplitter",
    "homology_aware": "biosieve.splitting.homology_aware:HomologyAwareSplitter",
    "random_kfold": "biosieve.splitting.random_kfold:RandomKFoldSplitter",
    "stratified_kfold": "biosieve.splitting.stratified_kfold:StratifiedKFoldSplitter",
    "group_kfold": "biosieve.splitting.group_kfold:GroupKFoldSplitter",
    "stratified_numeric_kfold": "biosieve.splitting.stratified_numeric_kfold:StratifiedNumericKFoldSplitter",
    "distance_aware_kfold": "biosieve.splitting.distance_aware_kfold:DistanceAwareKFoldSplitter",
}


def build_registry() -> StrategyRegistry:
    """Build and return the default reducer/splitter registry."""
    return StrategyRegistry(reducers=dict(REDUCERS), splitters=dict(SPLITTERS))
