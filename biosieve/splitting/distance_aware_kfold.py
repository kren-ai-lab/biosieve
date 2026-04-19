# ruff: noqa: ANN401, D102

"""Distance-aware k-fold splitter for out-of-distribution validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import (
    require_train_test_split,
    split_train_val,
    validate_kfold,
)
from biosieve.splitting.distance_aware import _dist_stats, _distance_to_centroid, build_distance_features

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class DistanceAwareKFoldSplitter:
    """Distance-ranked k-fold splitting."""

    feature_mode: str = "embeddings"
    metric: str = "cosine"
    n_splits: int = 5
    seed: int = 13
    shuffle_ties: bool = True
    val_size: float = 0.0
    drop_internal_index: bool = True
    embeddings_path: str | None = None
    ids_path: str | None = None
    descriptor_cols: list[str] | None = None
    descriptor_prefix: str | None = "desc_"
    standardize_descriptors: bool = True

    @property
    def strategy(self) -> str:
        return "distance_aware_kfold"

    def run_folds(self, df: pl.DataFrame, cols: Any) -> list[SplitResult]:
        validate_kfold(self.n_splits, self.val_size)
        work = df.with_row_index("__rowid")
        X, feat_idx, _meta = build_distance_features(
            work,
            cols,
            feature_mode=self.feature_mode,
            embeddings_path=self.embeddings_path or "embeddings.npy",
            ids_path=self.ids_path or "embedding_ids.csv",
            ids_col="id",
            descriptor_cols=self.descriptor_cols,
            descriptor_prefix=self.descriptor_prefix or "desc_",
            standardize=self.standardize_descriptors,
            dtype="float32",
        )
        d = _distance_to_centroid(X, self.metric)
        rng = np.random.default_rng(self.seed)
        key = -(d + rng.normal(0.0, 1e-12, size=d.shape[0])) if self.shuffle_ties else -d
        order = np.argsort(key)
        ranked = feat_idx[order]
        test_slices = np.array_split(ranked, self.n_splits)
        tts = (
            require_train_test_split("DistanceAwareKFoldSplitter with val_size > 0")
            if self.val_size > 0
            else None
        )
        folds: list[SplitResult] = []
        pos = {int(df_i): k for k, df_i in enumerate(feat_idx.tolist())}

        for fold_idx, test_idx in enumerate(test_slices):
            test_list = np.asarray(test_idx, dtype=int).tolist()
            test_set = set(test_list)
            train = work[[i for i in range(work.height) if i not in test_set]]
            test = work[test_list]
            val = None
            if self.val_size > 0:
                train, val = split_train_val(
                    train,
                    val_size=self.val_size,
                    seed=self.seed + fold_idx,
                    feature="DistanceAwareKFoldSplitter with val_size > 0",
                    train_test_split=tts,
                )

            train_rowids = train["__rowid"].to_list()
            test_rowids = test["__rowid"].to_list()
            train_d = np.asarray([d[pos[rid]] for rid in train_rowids if rid in pos], dtype=float)
            test_d = np.asarray([d[pos[rid]] for rid in test_rowids if rid in pos], dtype=float)

            if self.drop_internal_index:
                train = train.drop(["__rowid"])
                test = test.drop(["__rowid"])
                if val is not None:
                    val = val.drop(["__rowid"])

            folds.append(
                SplitResult(
                    train=train,
                    test=test,
                    val=val,
                    strategy=self.strategy,
                    params={"fold_index": fold_idx, "feature_mode": self.feature_mode},
                    stats={
                        "fold_index": fold_idx,
                        "n_total": work.height,
                        "n_train": train.height,
                        "n_test": test.height,
                        "n_val": val.height if val is not None else 0,
                        "distance_stats_global": _dist_stats(d),
                        "distance_stats_train": _dist_stats(train_d),
                        "distance_stats_test": _dist_stats(test_d),
                    },
                )
            )
        return folds
