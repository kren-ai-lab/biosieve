from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.reduction.backends.descriptor_backend import extract_descriptor_matrix, infer_descriptor_columns
from biosieve.reduction.backends.embedding_backend import load_embeddings
from biosieve.splitting.base import SplitResult
from biosieve.types import Columns
from biosieve.utils.logging import get_logger

log = get_logger(__name__)


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
    Compute distance to centroid for each row.

    Parameters
    ----------
    X:
        Feature matrix (n_samples, n_features).
    metric:
        Distance metric:
        - "cosine": 1 - cosine_similarity(x, centroid) in L2-normalized space
        - "euclidean": L2 distance to mean vector

    Returns
    -------
    np.ndarray
        Distance vector of shape (n_samples,).

    Raises
    ------
    ValueError
        If metric is not supported.
    """
    if metric == "cosine":
        Xn = _l2_normalize(X)
        c = Xn.mean(axis=0, keepdims=True)
        c = _l2_normalize(c)
        sims = (Xn * c).sum(axis=1)
        return 1.0 - sims
    if metric == "euclidean":
        c = X.mean(axis=0, keepdims=True)
        dif = X - c
        return np.sqrt((dif * dif).sum(axis=1))
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def _dist_stats(d: np.ndarray) -> Dict[str, Any]:
    if d.size == 0:
        return {
            "n": 0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "q25": None,
            "q75": None,
        }
    return {
        "n": int(d.size),
        "min": float(np.min(d)),
        "max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "std": float(np.std(d, ddof=1)) if d.size > 1 else 0.0,
        "median": float(np.median(d)),
        "q25": float(np.quantile(d, 0.25)),
        "q75": float(np.quantile(d, 0.75)),
    }


@dataclass(frozen=True)
class DistanceAwareSplitter:
    """
    Distance-aware split (OOD-oriented) selecting farthest samples as test (and optional val).

    This strategy selects the test set as the samples farthest from the centroid in a
    feature space (embeddings or descriptors). Optionally, validation can be selected
    either randomly from the remaining pool or as the "next farthest" samples.

    Supported feature modes
    -----------------------
    - feature_mode="embeddings":
        Uses embeddings exported as:
          - embeddings_path: .npy array (N, D)
          - ids_path: CSV with embedding row ids
        Distances typically use metric="cosine" (default).
    - feature_mode="descriptors":
        Uses numeric descriptor columns (explicit list or prefix selection).
        Distances typically use metric="euclidean", optionally after z-scoring.

    Selection policies
    ------------------
    - test_method="farthest" (v0.1):
        Take the top `n_test` farthest samples (among those with features).
    - val_method:
        - "random": pick val uniformly from the remaining candidates
        - "farthest_next": take the next `n_val` farthest samples after test

    Missing features behavior (embeddings mode)
    -------------------------------------------
    If some dataset ids do not have embeddings, they are *kept in TRAIN* by default.
    This is a conservative choice that avoids accidentally evaluating on samples that
    cannot be represented.

    Parameters
    ----------
    test_size:
        Fraction of full dataset assigned to test.
    val_size:
        Fraction of full dataset assigned to validation (0 disables validation).
    seed:
        Seed used for random val selection (val_method="random").
    feature_mode:
        "embeddings" or "descriptors".
    metric:
        "cosine" or "euclidean".
    embeddings_path, ids_path, ids_col:
        Embeddings layout parameters (embeddings mode).
    descriptor_prefix, descriptor_cols, standardize:
        Descriptor selection and normalization (descriptors mode).
    test_method:
        Only "farthest" is supported in v0.1.
    val_method:
        "random" or "farthest_next".
    dtype:
        Numpy dtype used to load/cast features ("float32" default).

    Returns
    -------
    SplitResult
        train/test/val DataFrames plus:
        - params: effective parameters and feature configuration
        - stats:
            - feature coverage
            - distance summaries (global/train/test/val) computed over *available-feature* pool
            - selection policy metadata

    Raises
    ------
    ValueError
        If split sizes are invalid, test/val sizes become zero after rounding, feature_mode
        or metric are invalid, no features are available, or descriptors contain NaNs.
    FileNotFoundError
        If embeddings files are missing (embeddings mode).
    ImportError
        If optional backends require missing dependencies (propagated).

    Notes
    -----
    - This strategy is designed for robustness/OOD evaluation: test is intentionally
      "harder" (farthest from centroid).
    - It does not enforce leakage constraints by itself (homology/structure/groups).
      If you need leakage-aware OOD splitting, a hybrid strategy is recommended.

    Examples
    --------
    Embeddings mode:

    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_distance_aware \\
    ...   --strategy distance_aware \\
    ...   --params params.yaml

    Descriptors mode:

    >>> biosieve split \\
    ...   --in dataset.csv \\
    ...   --outdir runs/split_distance_aware_desc \\
    ...   --strategy distance_aware \\
    ...   --params params.yaml
    """

    # Core split sizes
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13

    # Feature mode
    feature_mode: str = "embeddings"  # "embeddings" | "descriptors"

    # Metric
    metric: str = "cosine"  # "cosine" | "euclidean"

    # Embeddings inputs (when feature_mode="embeddings")
    embeddings_path: str = "embeddings.npy"
    ids_path: str = "embedding_ids.csv"
    ids_col: str = "id"

    # Descriptor inputs (when feature_mode="descriptors")
    descriptor_prefix: str = "desc_"
    descriptor_cols: Optional[List[str]] = None
    standardize: bool = True

    # Selection behavior
    test_method: str = "farthest"  # v0.1 only supports "farthest"
    val_method: str = "random"  # "random" | "farthest_next"

    # dtype
    dtype: str = "float32"

    @property
    def strategy(self) -> str:
        return "distance_aware"

    def _build_features(
        self, df: pd.DataFrame, cols: Columns
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Build feature matrix and return aligned dataframe indices.

        Returns
        -------
        X:
            Feature matrix for rows that have features.
        idx:
            Row indices in `df` corresponding to rows in X.
        meta:
            Feature coverage metadata.

        Raises
        ------
        ValueError
            If no usable features can be constructed.
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

            present_idx: List[int] = []
            emb_rows: List[int] = []
            missing = 0
            for i, sid in enumerate(ids):
                j = id_to_idx.get(sid)
                if j is None:
                    missing += 1
                    continue
                present_idx.append(i)
                emb_rows.append(j)

            if len(present_idx) == 0:
                raise ValueError(
                    "No dataset ids found in embedding ids file. Cannot run distance-aware split."
                )

            X = store.X[np.asarray(emb_rows, dtype=int)]
            idx = np.asarray(present_idx, dtype=int)
            meta = {
                "feature_mode": "embeddings",
                "n_total": int(len(df)),
                "n_with_features": int(len(idx)),
                "n_missing_features": int(missing),
                "coverage": float(len(idx) / len(df)) if len(df) else 0.0,
                "embeddings_path": self.embeddings_path,
                "ids_path": self.ids_path,
                "ids_col": self.ids_col,
                "n_features": int(X.shape[1]),
            }
            return X, idx, meta

        if self.feature_mode == "descriptors":
            dcols = infer_descriptor_columns(
                df, prefix=self.descriptor_prefix, explicit_cols=self.descriptor_cols
            )
            mat = extract_descriptor_matrix(df, dcols, dtype=self.dtype)
            X = mat.X
            idx = np.arange(len(df), dtype=int)
            meta = {
                "feature_mode": "descriptors",
                "n_total": int(len(df)),
                "n_with_features": int(len(df)),
                "n_missing_features": 0,
                "coverage": 1.0,
                "descriptor_prefix": self.descriptor_prefix,
                "descriptor_cols_used_preview": dcols[:10] + (["..."] if len(dcols) > 10 else []),
                "n_descriptors": int(len(dcols)),
                "standardize": bool(self.standardize),
                "n_features": int(X.shape[1]),
            }
            return X, idx, meta

        raise ValueError("feature_mode must be 'embeddings' or 'descriptors'")

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:

        log.info("distance_aware:start | metric=%s | test_size=%.3f", self.metric, self.test_size)
        log.debug("distance_aware:params | %s", self.__dict__)

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

        # Standardize descriptors if requested (useful mainly for euclidean)
        z_mu = None
        z_sd = None
        if self.feature_mode == "descriptors" and self.standardize:
            X, z_mu, z_sd = _zscore_fit_transform(X)

        # Compute distance-to-centroid for candidates with features
        dist = _distance_to_centroid(X, metric=self.metric)

        # Sort candidates by distance desc
        order = np.argsort(-dist, kind="mergesort")
        ranked_df_idx = feat_idx[order]  # indices in work, sorted by farthest-first
        ranked_dist = dist[order]

        # Choose test indices (only among feature-covered candidates)
        if len(ranked_df_idx) < n_test:
            raise ValueError(
                f"Not enough feature-covered samples to allocate test. "
                f"need n_test={n_test}, have n_candidates={len(ranked_df_idx)}"
            )
        test_idx = ranked_df_idx[:n_test].tolist()

        # Choose val indices
        remaining_idx = ranked_df_idx[n_test:].tolist()
        remaining_dist = ranked_dist[n_test:]

        if n_val > 0:
            if len(remaining_idx) < n_val:
                raise ValueError("Not enough remaining feature-covered samples to allocate validation.")
            if self.val_method == "farthest_next":
                val_idx = remaining_idx[:n_val]
            else:
                rng = np.random.default_rng(self.seed)
                sel = rng.choice(np.asarray(remaining_idx, dtype=int), size=n_val, replace=False)
                val_idx = sel.tolist()
        else:
            val_idx = []

        test_set = set(int(i) for i in test_idx)
        val_set = set(int(i) for i in val_idx)

        # TRAIN gets everything not in test/val (including missing-feature rows)
        train_idx = [i for i in range(n) if i not in test_set and i not in val_set]

        train = work.iloc[train_idx].reset_index(drop=True)
        test = work.iloc[test_idx].reset_index(drop=True)
        val = work.iloc[val_idx].reset_index(drop=True) if n_val > 0 else None

        # Distance stats are computed only for samples WITH FEATURES
        # We can still compute per-split stats by intersecting indices with feat_idx.
        feat_idx_set = set(int(i) for i in feat_idx.tolist())

        def _subset_dist(idxs: List[int]) -> np.ndarray:
            # idxs are dataframe row indices; select those that have features
            mask = [i for i in idxs if int(i) in feat_idx_set]
            if not mask:
                return np.asarray([], dtype=float)
            # map df indices -> position in feat_idx
            pos = {int(df_i): k for k, df_i in enumerate(feat_idx.tolist())}
            return dist[np.asarray([pos[int(i)] for i in mask], dtype=int)]

        d_train = _subset_dist(train_idx)
        d_test = _subset_dist(test_idx)
        d_val = _subset_dist(val_idx) if n_val > 0 else np.asarray([], dtype=float)

        stats: Dict[str, Any] = {
            "n_total": int(n),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_val": int(len(val)) if val is not None else 0,
            "feature_meta": feat_meta,
            "metric": self.metric,
            "test_method": self.test_method,
            "val_method": self.val_method,
            "distance_stats_global": _dist_stats(dist),
            "distance_stats_train": _dist_stats(d_train),
            "distance_stats_test": _dist_stats(d_test),
            "distance_stats_val": _dist_stats(d_val) if n_val > 0 else None,
            "note": "test is selected as farthest-from-centroid in feature space (distance-aware).",
        }

        log.info(
            "distance_aware:stats | train=%d | val=%d | test=%d",
            stats["n_train"],
            stats["n_val"],
            stats["n_test"],
        )

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
