from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult

from biosieve.reduction.backends.embedding_backend import load_embeddings
from biosieve.reduction.backends.descriptor_backend import infer_descriptor_columns, extract_descriptor_matrix


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 <= val_size < 1.0):
        raise ValueError("val_size must be in [0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")


def _zscore_fit_transform(X: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    Xz = (X - mu) / sd
    return Xz, mu, sd


def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)


def _distance_to_centroid(X: np.ndarray, metric: str) -> np.ndarray:
    """
    Returns distance to centroid for each row.
      - cosine: 1 - cos_sim(x, c) where both normalized
      - euclidean: ||x - mean||_2
    """
    if metric == "cosine":
        Xn = _l2_normalize(X)
        c = Xn.mean(axis=0, keepdims=True)
        c = _l2_normalize(c)
        sims = (Xn * c).sum(axis=1)
        return 1.0 - sims
    elif metric == "euclidean":
        c = X.mean(axis=0, keepdims=True)
        dif = X - c
        return np.sqrt((dif * dif).sum(axis=1))
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")


@dataclass(frozen=True)
class DistanceAwareSplitter:
    """
    Distance-aware split (v0.1):
      Select test (and optional val) as the samples farthest from the centroid
      in a feature space.

    Supports:
      - embeddings via (embeddings.npy + embedding_ids.csv) with cosine distance
      - descriptors via columns (desc_*) with euclidean distance (optionally z-scored)

    Selection:
      - test_method: 'farthest' (default)   -> top-N farthest points as test
      - val_method:  'random' or 'farthest_next'
          random: pick val randomly from remaining
          farthest_next: next farthest points after test

    Missing features:
      - if some ids have no embedding (embeddings mode), they are kept in TRAIN (safe default)
    """

    # Core split sizes
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    # Feature mode
    feature_mode: str = "embeddings"  # "embeddings" | "descriptors"

    # Metric
    metric: str = "cosine"            # "cosine" | "euclidean"

    # Embeddings inputs (when feature_mode="embeddings")
    embeddings_path: str = "embeddings.npy"
    ids_path: str = "embedding_ids.csv"
    ids_col: str = "id"

    # Descriptor inputs (when feature_mode="descriptors")
    descriptor_prefix: str = "desc_"
    descriptor_cols: Optional[List[str]] = None
    standardize: bool = True

    # Selection behavior
    test_method: str = "farthest"     # v0.1 only supports "farthest"
    val_method: str = "random"        # "random" | "farthest_next"

    # dtype
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        return "distance_aware"

    def _build_features(self, df: pd.DataFrame, cols: Columns) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Return:
          X: features for rows that have features
          idx: row indices in df that correspond to X
          meta: stats (coverage, etc.)
        """
        ids = df[cols.id_col].astype(str).tolist()

        if self.feature_mode == "embeddings":
            store = load_embeddings(
                embeddings_path=self.embeddings_path,
                ids_path=self.ids_path,
                ids_col=self.ids_col,
                dtype=self.dtype,
            )
            id_to_idx = {sid: i for i, sid in enumerate(store.ids)}

            present_idx = []
            emb_rows = []
            missing = 0
            for i, sid in enumerate(ids):
                j = id_to_idx.get(sid)
                if j is None:
                    missing += 1
                    continue
                present_idx.append(i)
                emb_rows.append(j)

            if len(present_idx) == 0:
                raise ValueError("No dataset ids found in embedding ids file. Cannot run distance-aware split.")

            X = store.X[np.array(emb_rows, dtype=int)]
            idx = np.array(present_idx, dtype=int)
            meta = {
                "feature_mode": "embeddings",
                "n_total": len(df),
                "n_with_features": int(len(idx)),
                "n_missing_features": int(missing),
                "coverage": float(len(idx) / len(df)) if len(df) else 0.0,
                "embeddings_path": self.embeddings_path,
                "ids_path": self.ids_path,
                "ids_col": self.ids_col,
            }
            return X, idx, meta

        elif self.feature_mode == "descriptors":
            dcols = infer_descriptor_columns(df, prefix=self.descriptor_prefix, explicit_cols=self.descriptor_cols)
            mat = extract_descriptor_matrix(df, dcols, dtype=self.dtype)
            X = mat.X
            idx = np.arange(len(df), dtype=int)
            meta = {
                "feature_mode": "descriptors",
                "n_total": len(df),
                "n_with_features": int(len(df)),
                "n_missing_features": 0,
                "coverage": 1.0,
                "descriptor_prefix": self.descriptor_prefix,
                "descriptor_cols_used": dcols[:10] + (["..."] if len(dcols) > 10 else []),
                "n_descriptors": len(dcols),
                "standardize": bool(self.standardize),
            }
            return X, idx, meta

        else:
            raise ValueError("feature_mode must be 'embeddings' or 'descriptors'")

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)

        n = len(work)
        n_test = int(round(n * self.test_size))
        n_val = int(round(n * self.val_size)) if self.val_size > 0 else 0
        n_train = n - n_test - n_val
        if n_train <= 0:
            raise ValueError("Split sizes leave no training samples. Reduce test_size/val_size.")
        if n_test <= 0:
            raise ValueError("test_size too small -> no test samples after rounding. Increase test_size.")
        if self.val_size > 0 and n_val <= 0:
            raise ValueError("val_size too small -> no validation samples after rounding. Increase val_size.")

        if self.test_method != "farthest":
            raise ValueError("v0.1 supports test_method='farthest' only")
        if self.val_method not in {"random", "farthest_next"}:
            raise ValueError("val_method must be 'random' or 'farthest_next'")
        if self.metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        X, feat_idx, feat_meta = self._build_features(work, cols)

        # Standardize descriptors if requested (only makes sense for euclidean / descriptors)
        z_mu = None
        z_sd = None
        if self.feature_mode == "descriptors" and self.standardize:
            X, z_mu, z_sd = _zscore_fit_transform(X)

        # Compute distance-to-centroid for candidates with features
        dist = _distance_to_centroid(X, metric=self.metric)

        # Sort feature-rows by distance desc
        order = np.argsort(-dist, kind="mergesort")
        ranked_df_idx = feat_idx[order]  # indices in work, sorted by "most different first"

        # Choose test indices
        test_idx = ranked_df_idx[:n_test].tolist()

        # Choose val indices
        remaining = ranked_df_idx[n_test:].tolist()

        if n_val > 0:
            if self.val_method == "farthest_next":
                val_idx = remaining[:n_val]
                remaining = remaining[n_val:]
            else:
                rng = np.random.default_rng(self.seed)
                if len(remaining) < n_val:
                    raise ValueError("Not enough remaining samples to allocate validation.")
                sel = rng.choice(np.array(remaining, dtype=int), size=n_val, replace=False)
                val_idx = sel.tolist()
                remaining_set = set(remaining)
                for s in val_idx:
                    remaining_set.discard(int(s))
                remaining = list(remaining_set)
        else:
            val_idx = []

        # TRAIN gets:
        #  - all samples not in test/val
        #  - plus any samples missing features (in embeddings mode) will naturally land here
        test_set = set(test_idx)
        val_set = set(val_idx)

        train_idx = [i for i in range(n) if i not in test_set and i not in val_set]

        train = work.iloc[train_idx].reset_index(drop=True)
        test = work.iloc[test_idx].reset_index(drop=True)
        val = work.iloc[val_idx].reset_index(drop=True) if n_val > 0 else None

        stats: Dict[str, Any] = {
            "n_total": int(n),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "feature_meta": feat_meta,
            "metric": self.metric,
            "test_method": self.test_method,
            "val_method": self.val_method,
            "note": "test is selected as farthest-from-centroid in feature space (distance-aware v0.1).",
        }

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={
                "test_size": self.test_size,
                "val_size": self.val_size,
                "seed": self.seed,
                "feature_mode": self.feature_mode,
                "metric": self.metric,
                "embeddings_path": self.embeddings_path,
                "ids_path": self.ids_path,
                "ids_col": self.ids_col,
                "descriptor_prefix": self.descriptor_prefix,
                "descriptor_cols": self.descriptor_cols,
                "standardize": self.standardize,
                "test_method": self.test_method,
                "val_method": self.val_method,
                "dtype": self.dtype,
                "zscore_used": bool(z_mu is not None),
            },
            stats=stats,
        )
