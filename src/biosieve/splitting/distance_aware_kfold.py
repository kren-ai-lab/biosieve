from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult


def _try_import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        return train_test_split
    except Exception:
        return None


def _label_stats_from_array(x: np.ndarray) -> Dict[str, Any]:
    if x.size == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


def _standardize(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (X - mu) / sd


def _cosine_distance_to_centroid(X: np.ndarray) -> np.ndarray:
    # centroid then normalize
    c = X.mean(axis=0, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    sim = (Xn @ cn.T).reshape(-1)
    return 1.0 - sim


def _euclidean_distance_to_centroid(X: np.ndarray) -> np.ndarray:
    c = X.mean(axis=0, keepdims=True)
    return np.linalg.norm(X - c, axis=1)


def _load_embedding_ids(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ids_path not found: {p}")

    df = pd.read_csv(p)
    # accept common variants
    for col in ["id", "ids", "sequence_id", "uniprot_id"]:
        if col in df.columns:
            return df[col].astype(str).tolist()

    # fallback: 1-column CSV
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()

    raise ValueError(
        f"ids_path must contain a recognizable id column (id/ids/sequence_id/uniprot_id) "
        f"or be a single-column CSV. Found: {df.columns.tolist()}"
    )


def _load_features_embeddings(df: pd.DataFrame, cols: Columns, embeddings_path: str, ids_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ep = Path(embeddings_path)
    if not ep.exists():
        raise FileNotFoundError(f"embeddings_path not found: {ep}")

    X = np.load(ep)  # (N, D)
    ids = _load_embedding_ids(ids_path)
    if len(ids) != X.shape[0]:
        raise ValueError(
            f"Mismatch embeddings rows vs ids. embeddings.shape[0]={X.shape[0]} vs len(ids)={len(ids)}"
        )

    id_to_idx = {str(i): k for k, i in enumerate(ids)}

    df_ids = df[cols.id_col].astype(str).tolist()
    idx = []
    missing = 0
    for sid in df_ids:
        k = id_to_idx.get(sid)
        if k is None:
            missing += 1
            idx.append(-1)
        else:
            idx.append(k)

    if missing > 0:
        raise ValueError(
            f"Missing embeddings for {missing}/{len(df_ids)} samples. "
            "Provide full embeddings coverage or pre-filter the dataset."
        )

    idx_arr = np.asarray(idx, dtype=int)
    return X[idx_arr, :], idx_arr


def _load_features_descriptors(
    df: pd.DataFrame,
    *,
    descriptor_cols: Optional[List[str]],
    descriptor_prefix: Optional[str],
    standardize: bool,
) -> Tuple[np.ndarray, List[str]]:
    if descriptor_cols is None:
        if not descriptor_prefix:
            raise ValueError("For descriptors mode, provide descriptor_cols or descriptor_prefix.")
        cols = [c for c in df.columns if c.startswith(descriptor_prefix)]
    else:
        cols = list(descriptor_cols)

    if len(cols) == 0:
        raise ValueError("No descriptor columns selected for descriptors mode.")

    Xdf = df[cols].copy()

    # coerce to numeric
    for c in cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    if Xdf.isna().any().any():
        bad = Xdf.isna().sum().sum()
        raise ValueError(f"Descriptor matrix contains NaNs after coercion ({bad} NaN cells). Clean/impute first.")

    X = Xdf.to_numpy(dtype=float)
    if standardize:
        X = _standardize(X)
    return X, cols


@dataclass(frozen=True)
class DistanceAwareKFoldSplitter:
    """
    Distance-aware K-Fold splitting (OOD-oriented CV).

    This splitter builds folds by ranking samples according to their distance
    to the global centroid in feature space, then partitioning that ranking into
    disjoint test folds.

    Feature modes
    -------------
    - embeddings: uses an external embeddings matrix aligned by ids
    - descriptors: uses numeric columns from the input CSV

    Metrics
    -------
    - cosine: 1 - cosine_similarity(x, centroid)
    - euclidean: ||x - centroid||

    Parameters
    ----------
    feature_mode:
        "embeddings" or "descriptors".
    metric:
        "cosine" or "euclidean".
    n_splits:
        Number of folds (test folds are disjoint and cover the dataset).
    shuffle_ties:
        If True, breaks distance ties using a seeded random jitter.
    seed:
        Seed used for tie-breaking and optional val sampling.
    val_size:
        Optional validation fraction sampled randomly from each fold's train set.

    Embeddings mode parameters
    -------------------------
    embeddings_path:
        Path to .npy file with shape (N, D).
    ids_path:
        CSV with embedding row ids.

    Descriptors mode parameters
    ---------------------------
    descriptor_cols:
        Explicit descriptor columns list. If None, uses descriptor_prefix.
    descriptor_prefix:
        Select columns starting with this prefix (e.g., "desc_").
    standardize_descriptors:
        If True, z-score descriptors before distances.

    Notes
    -----
    - This is not a "random CV": it intentionally creates OOD test folds.
    - Useful to evaluate model robustness / generalization to distant regions.
    """

    feature_mode: str = "embeddings"    # "embeddings" | "descriptors"
    metric: str = "cosine"             # "cosine" | "euclidean"

    n_splits: int = 5
    seed: int = 13
    shuffle_ties: bool = True

    val_size: float = 0.0

    # embeddings
    embeddings_path: Optional[str] = None
    ids_path: Optional[str] = None

    # descriptors
    descriptor_cols: Optional[List[str]] = None
    descriptor_prefix: Optional[str] = "desc_"
    standardize_descriptors: bool = True

    @property
    def strategy(self) -> str:
        return "distance_aware_kfold"

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> List[SplitResult]:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0.0 <= self.val_size < 1.0):
            raise ValueError("val_size must be in [0, 1)")

        work = df.copy().reset_index(drop=True)
        n = len(work)
        if n < self.n_splits:
            raise ValueError(f"Not enough samples (n={n}) for n_splits={self.n_splits}")

        # --- Load features ---
        feature_info: Dict[str, Any] = {"feature_mode": self.feature_mode}

        if self.feature_mode == "embeddings":
            if not self.embeddings_path or not self.ids_path:
                raise ValueError("Embeddings mode requires embeddings_path and ids_path.")
            X, idx_arr = _load_features_embeddings(work, cols, self.embeddings_path, self.ids_path)
            feature_info.update({"embeddings_path": self.embeddings_path, "ids_path": self.ids_path, "n_features": int(X.shape[1])})

        elif self.feature_mode == "descriptors":
            X, used_cols = _load_features_descriptors(
                work,
                descriptor_cols=self.descriptor_cols,
                descriptor_prefix=self.descriptor_prefix,
                standardize=self.standardize_descriptors,
            )
            feature_info.update(
                {
                    "descriptor_cols": used_cols,
                    "standardize_descriptors": bool(self.standardize_descriptors),
                    "n_features": int(X.shape[1]),
                }
            )
        else:
            raise ValueError("feature_mode must be 'embeddings' or 'descriptors'")

        # --- Distances ---
        if self.metric == "cosine":
            d = _cosine_distance_to_centroid(X)
        elif self.metric == "euclidean":
            d = _euclidean_distance_to_centroid(X)
        else:
            raise ValueError("metric must be 'cosine' or 'euclidean'")

        # --- Rank samples farthest -> closest ---
        rng = np.random.default_rng(self.seed)
        if self.shuffle_ties:
            # tiny jitter to break ties deterministically
            jitter = rng.normal(0.0, 1e-12, size=d.shape[0])
            key = -(d + jitter)
        else:
            key = -d
        order = np.argsort(key)

        # Split ranking into disjoint test folds
        test_slices = np.array_split(order, self.n_splits)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                raise ImportError(
                    "val_size > 0 requires scikit-learn. Install: conda install -c conda-forge scikit-learn"
                )

        folds: List[SplitResult] = []

        all_idx = np.arange(n, dtype=int)

        for fold_idx, test_idx in enumerate(test_slices):
            test_idx = np.asarray(test_idx, dtype=int)

            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False
            train_idx = all_idx[mask]

            train_df = work.iloc[train_idx].copy().reset_index(drop=True)
            test_df = work.iloc[test_idx].copy().reset_index(drop=True)

            # optional val
            val_df: Optional[pd.DataFrame] = None
            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)
                train_df, val_df = tts(
                    train_df,
                    test_size=self.val_size,
                    random_state=seed_fold,
                    shuffle=True,
                    stratify=None,
                )
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)

            # distance stats
            d_train = d[train_idx]
            d_test = d[test_idx]

            stats: Dict[str, Any] = {
                "fold_index": int(fold_idx),
                "n_total": int(len(df)),
                "n_used": int(n),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_val": int(len(val_df)) if val_df is not None else 0,

                "metric": self.metric,
                "distance_stats_global": _label_stats_from_array(d),
                "distance_stats_train": _label_stats_from_array(d_train),
                "distance_stats_test": _label_stats_from_array(d_test),
            }

            if val_df is not None:
                # val distances: need indices in original fold train; approximate by recomputing from ids is heavy,
                # but we can compute them by mapping via original index values before reset.
                # Simpler: compute d_val using original indices from the selection.
                # We can recover original indices by using train_idx split before reset:
                # (val_df was sampled from train_df after reset, so we can't index d directly).
                # For now we provide val stats as None; if you want exact val stats, we can enhance runner to keep original indices.
                stats["distance_stats_val"] = None
                stats["note_val_distances"] = "val distances not computed in v0.1 (requires index tracking)."

            folds.append(
                SplitResult(
                    train=train_df,
                    test=test_df,
                    val=val_df,
                    strategy=self.strategy,
                    params={
                        "feature_mode": self.feature_mode,
                        "metric": self.metric,
                        "n_splits": self.n_splits,
                        "seed": self.seed,
                        "shuffle_ties": self.shuffle_ties,
                        "val_size": self.val_size,
                        **feature_info,
                        "fold_index": fold_idx,
                    },
                    stats=stats,
                )
            )

        return folds
