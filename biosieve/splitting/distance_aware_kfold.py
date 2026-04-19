# ruff: noqa: ANN202, ANN401, D102, EM101, PLC0415, SLF001, TC002, TRY003, TRY300

"""Distance-aware k-fold splitter for out-of-distribution validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.splitting.distance_aware import DistanceAwareSplitter, _dist_stats


def _try_import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split

        return train_test_split
    except ImportError:
        return None


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
        helper = DistanceAwareSplitter(
            feature_mode=self.feature_mode,
            metric=self.metric,
            embeddings_path=self.embeddings_path or "embeddings.npy",
            ids_path=self.ids_path or "embedding_ids.csv",
            descriptor_cols=self.descriptor_cols,
            descriptor_prefix=self.descriptor_prefix or "desc_",
            standardize=self.standardize_descriptors,
            seed=self.seed,
        )
        work = df.with_row_index("__rowid")
        X, feat_idx, _meta = helper._build_features(work, cols)
        from biosieve.splitting.distance_aware import _distance_to_centroid

        d = _distance_to_centroid(X, self.metric)
        rng = np.random.default_rng(self.seed)
        key = -(d + rng.normal(0.0, 1e-12, size=d.shape[0])) if self.shuffle_ties else -d
        order = np.argsort(key)
        ranked = feat_idx[order]
        test_slices = np.array_split(ranked, self.n_splits)
        tts = _try_import_train_test_split() if self.val_size > 0 else None
        folds: list[SplitResult] = []
        pos = {int(df_i): k for k, df_i in enumerate(feat_idx.tolist())}

        for fold_idx, test_idx in enumerate(test_slices):
            test_list = np.asarray(test_idx, dtype=int).tolist()
            test_set = set(test_list)
            train = work[[i for i in range(work.height) if i not in test_set]]
            test = work[test_list]
            val = None
            if self.val_size > 0:
                if tts is None:
                    raise ImportError("val_size > 0 requires scikit-learn.")
                inner_idx = np.arange(train.height)
                train_keep_idx, val_idx = tts(
                    inner_idx,
                    test_size=self.val_size,
                    random_state=self.seed + fold_idx,
                    shuffle=True,
                    stratify=None,
                )
                val = train[val_idx]
                train = train[train_keep_idx]

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
