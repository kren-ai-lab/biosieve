# ruff: noqa: ANN401, D102, EM101, TRY003

"""Distance-aware split strategy targeting out-of-distribution evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from biosieve.reduction.backends.descriptor_backend import extract_descriptor_matrix, infer_descriptor_columns
from biosieve.reduction.backends.embedding_backend import load_embeddings
from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

log = get_logger(__name__)
DESCRIPTOR_PREVIEW_LIMIT = 10


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def _distance_to_centroid(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        Xn = _l2_normalize(X)
        c = _l2_normalize(Xn.mean(axis=0, keepdims=True))
        return 1.0 - (Xn * c).sum(axis=1)
    if metric == "euclidean":
        c = X.mean(axis=0, keepdims=True)
        return np.linalg.norm(X - c, axis=1)
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def _dist_stats(d: np.ndarray) -> dict[str, Any]:
    if d.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None}
    return {
        "n": int(d.size),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "std": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
    }


@dataclass(frozen=True)
class DistanceAwareSplitter:
    """Distance-aware split using farthest-from-centroid samples as test."""

    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13
    feature_mode: str = "embeddings"
    metric: str = "cosine"
    embeddings_path: str = "embeddings.npy"
    ids_path: str = "embedding_ids.csv"
    ids_col: str = "id"
    descriptor_prefix: str = "desc_"
    descriptor_cols: list[str] | None = None
    standardize: bool = True
    test_method: str = "farthest"
    val_method: str = "random"
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        return "distance_aware"

    def _build_features(self, df: pl.DataFrame, cols: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        ids = df[cols.id_col].cast(pl.String).to_list()
        if self.feature_mode == "embeddings":
            store = load_embeddings(self.embeddings_path, self.ids_path, self.ids_col, self.dtype)
            id_to_idx = {sid: i for i, sid in enumerate(store.ids)}
            present_idx: list[int] = []
            emb_rows: list[int] = []
            for i, sid in enumerate(ids):
                if sid in id_to_idx:
                    present_idx.append(i)
                    emb_rows.append(id_to_idx[sid])
            if not present_idx:
                raise ValueError("No dataset ids found in embedding ids file.")
            return (
                store.X[np.asarray(emb_rows, dtype=int)],
                np.asarray(present_idx, dtype=int),
                {"feature_mode": "embeddings", "n_with_features": len(present_idx), "n_total": df.height},
            )
        if self.feature_mode == "descriptors":
            dcols = infer_descriptor_columns(
                df, prefix=self.descriptor_prefix, explicit_cols=self.descriptor_cols
            )
            X = extract_descriptor_matrix(df, dcols, self.dtype).X
            if self.standardize:
                mu = X.mean(axis=0, keepdims=True)
                sd = np.where(X.std(axis=0, keepdims=True) == 0.0, 1.0, X.std(axis=0, keepdims=True))
                X = (X - mu) / sd
            return (
                X,
                np.arange(df.height, dtype=int),
                {
                    "feature_mode": "descriptors",
                    "descriptor_cols_used_preview": dcols[:DESCRIPTOR_PREVIEW_LIMIT],
                    "n_total": df.height,
                    "n_with_features": df.height,
                },
            )
        raise ValueError("feature_mode must be 'embeddings' or 'descriptors'")

    def run(self, df: pl.DataFrame, cols: Any) -> SplitResult:
        _validate_sizes(self.test_size, self.val_size)
        work = df.clone()
        n = work.height
        n_test = round(n * self.test_size)
        n_val = round(n * self.val_size) if self.val_size > 0 else 0
        if n - n_test - n_val <= 0 or n_test <= 0:
            raise ValueError("Split sizes leave no valid train/test partition.")

        X, feat_idx, feat_meta = self._build_features(work, cols)
        n_feat = int(feat_idx.size)
        if n_feat < n_test:
            msg = f"Not enough feature-covered samples to allocate test split: need {n_test}, found {n_feat}."
            raise ValueError(msg)
        if n_val > 0 and n_feat < n_test + n_val:
            msg = (
                "Not enough feature-covered samples to allocate test and validation splits: "
                f"need {n_test + n_val}, found {n_feat}."
            )
            raise ValueError(msg)
        dist = _distance_to_centroid(X, self.metric)
        order = feat_idx[np.argsort(-dist, kind="mergesort")]
        test_idx = order[:n_test].tolist()
        remaining = order[n_test:].tolist()

        if n_val > 0:
            if self.val_method == "farthest_next":
                val_idx = remaining[:n_val]
            else:
                rng = np.random.default_rng(self.seed)
                val_idx = rng.choice(np.asarray(remaining, dtype=int), size=n_val, replace=False).tolist()
        else:
            val_idx = []

        test_set = set(map(int, test_idx))
        val_set = set(map(int, val_idx))
        train_idx = [i for i in range(n) if i not in test_set and i not in val_set]

        train = work[train_idx]
        test = work[test_idx]
        val = work[val_idx] if val_idx else None

        pos = {int(df_i): k for k, df_i in enumerate(feat_idx.tolist())}

        def _subset_dist(idxs: list[int]) -> np.ndarray:
            picked = [pos[i] for i in idxs if i in pos]
            return dist[np.asarray(picked, dtype=int)] if picked else np.asarray([], dtype=float)

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={"feature_mode": self.feature_mode, "metric": self.metric},
            stats={
                "n_total": n,
                "n_train": train.height,
                "n_test": test.height,
                "n_val": val.height if val is not None else 0,
                "feature_meta": feat_meta,
                "distance_stats_global": _dist_stats(dist),
                "distance_stats_train": _dist_stats(_subset_dist(train_idx)),
                "distance_stats_test": _dist_stats(_subset_dist(test_idx)),
            },
        )
