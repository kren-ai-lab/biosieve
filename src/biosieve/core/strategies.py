from __future__ import annotations

from biosieve.core.registry import StrategyRegistry
from biosieve.reduction import (ExactDedupReducer, IdentityGreedyReducer, KmerJaccardReducer,
                                MMseqs2Reducer, EmbeddingCosineReducer, DescriptorEuclideanReducer,
                                StructuralDistanceReducer)

from biosieve.splitting import (RandomSplitter, StratifiedSplitter, 
                                GroupSplitter, TimeSplitter, DistanceAwareSplitter,
                                HomologyAwareSplitter, ClusterAwareSplitter, StratifiedNumericSplitter,
                                RandomKFoldSplitter, StratifiedKFoldSplitter, GroupKFoldSplitter,
                                StratifiedNumericKFoldSplitter, DistanceAwareKFoldSplitter)

from biosieve.core.registry import StrategyRegistry
from biosieve.core.spec import StrategySpec

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
        "group": GroupSplitter,
        "time": TimeSplitter,
        "distance_aware": DistanceAwareSplitter,
        "homology_aware" : HomologyAwareSplitter,
        "cluster_aware" : ClusterAwareSplitter,
        "stratified_numeric" : StratifiedNumericSplitter,
        "random_kfold": RandomKFoldSplitter,
        "stratified_kfold" : StratifiedKFoldSplitter,
        "group_kfold" : GroupKFoldSplitter,
        "stratified_numeric_kfold" : StratifiedNumericKFoldSplitter,
        "distance_aware_kfold" : DistanceAwareKFoldSplitter
    }
    
    return StrategyRegistry(reducers=reducers, splitters=splitters)

def build_registry_light() -> StrategyRegistry:
    """
    Build a light registry using lazy StrategySpec entries.

    This registry is safe for `biosieve info` and `biosieve validate` because it
    does not import optional heavy dependencies at import time.
    """
    reg = StrategyRegistry()

    # ---- reducers ----
    reg.add_reducer("exact", StrategySpec("exact", "reducer", "biosieve.reduction.exact:ExactDedupReducer"))
    reg.add_reducer("kmer_jaccard", StrategySpec("kmer_jaccard", "reducer", "biosieve.reduction.kmer_jaccard:KmerJaccardReducer"))
    reg.add_reducer("identity_greedy", StrategySpec("identity_greedy", "reducer", "biosieve.reduction.identity_greedy:IdentityGreedyReducer"))

    reg.add_reducer("mmseqs2", StrategySpec("mmseqs2", "reducer", "biosieve.reduction.mmseqs2:MMseqs2Reducer"))
    reg.add_reducer("embedding_cosine", StrategySpec("embedding_cosine", "reducer", "biosieve.reduction.embedding_cosine:EmbeddingCosineReducer"))
    reg.add_reducer("descriptor_euclidean", StrategySpec("descriptor_euclidean", "reducer", "biosieve.reduction.descriptor_euclidean:DescriptorEuclideanReducer"))
    reg.add_reducer("structural_distance", StrategySpec("structural_distance", "reducer", "biosieve.reduction.structural_distance:StructuralDistanceReducer"))

    # ---- splitters ----
    reg.add_splitter("random", StrategySpec("random", "splitter", "biosieve.splitting.random:RandomSplitter"))
    reg.add_splitter("stratified", StrategySpec("stratified", "splitter", "biosieve.splitting.stratified:StratifiedSplitter"))
    reg.add_splitter("stratified_numeric", StrategySpec("stratified_numeric", "splitter", "biosieve.splitting.stratified_numeric:StratifiedNumericSplitter"))
    reg.add_splitter("group", StrategySpec("group", "splitter", "biosieve.splitting.group:GroupSplitter"))
    reg.add_splitter("time", StrategySpec("time", "splitter", "biosieve.splitting.time_based:TimeBasedSplitter"))
    reg.add_splitter("distance_aware", StrategySpec("distance_aware", "splitter", "biosieve.splitting.distance_aware:DistanceAwareSplitter"))
    reg.add_splitter("cluster_aware", StrategySpec("cluster_aware", "splitter", "biosieve.splitting.cluster:ClusterAwareSplitter"))
    reg.add_splitter("homology_aware", StrategySpec("homology_aware", "splitter", "biosieve.splitting.homology_aware:HomologyAwareSplitter"))

    # kfold family
    reg.add_splitter("group_kfold", StrategySpec("group_kfold", "splitter", "biosieve.splitting.group_kfold:GroupKFoldSplitter"))
    reg.add_splitter("stratified_numeric_kfold", StrategySpec("stratified_numeric_kfold", "splitter", "biosieve.splitting.stratified_numeric_kfold:StratifiedNumericKFoldSplitter"))
    reg.add_splitter("distance_aware_kfold", StrategySpec("distance_aware_kfold", "splitter", "biosieve.splitting.distance_aware_kfold:DistanceAwareKFoldSplitter"))

    return reg