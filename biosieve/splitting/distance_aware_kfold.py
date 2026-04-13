"""Distance-aware k-fold splitter for out-of-distribution validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)

_INTERNAL_IDX_COL = "_biosieve_row_idx__"
MIN_KFOLD_SPLITS = 2


class _TrainTestSplitFn(Protocol):
    def __call__(
        self,
        df: pd.DataFrame,
        *,
        test_size: float,
        random_state: int,
        shuffle: bool,
        stratify: None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        return cast("_TrainTestSplitFn", train_test_split)
    except ImportError:
        return None


def _label_stats_from_array(x: np.ndarray) -> dict[str, Any]:
    if x.size == 0:
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
    c = X.mean(axis=0, keepdims=True)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    sim = (Xn @ cn.T).reshape(-1)
    return 1.0 - sim


def _euclidean_distance_to_centroid(X: np.ndarray) -> np.ndarray:
    c = X.mean(axis=0, keepdims=True)
    return np.linalg.norm(X - c, axis=1)


def _load_embedding_ids(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        msg = f"ids_path not found: {p}"
        raise FileNotFoundError(msg)

    df = pd.read_csv(p)
    for col in ["id", "ids", "sequence_id", "uniprot_id"]:
        if col in df.columns:
            return df[col].astype(str).tolist()

    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(str).tolist()

    msg = (
        "ids_path must contain a recognizable id column (id/ids/sequence_id/uniprot_id) "
        f"or be a single-column CSV. Found: {df.columns.tolist()}"
    )
    raise ValueError(
        msg
    )


def _load_features_embeddings(
    df: pd.DataFrame,
    cols: Columns,
    embeddings_path: str,
    ids_path: str,
) -> np.ndarray:
    ep = Path(embeddings_path)
    if not ep.exists():
        msg = f"embeddings_path not found: {ep}"
        raise FileNotFoundError(msg)

    X = np.load(ep)  # (N, D)
    ids = _load_embedding_ids(ids_path)
    if len(ids) != X.shape[0]:
        msg = f"Mismatch embeddings rows vs ids. embeddings.shape[0]={X.shape[0]} vs len(ids)={len(ids)}"
        raise ValueError(
            msg
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
        msg = (
            f"Missing embeddings for {missing}/{len(df_ids)} samples. "
            "Provide full embeddings coverage or pre-filter the dataset."
        )
        raise ValueError(
            msg
        )

    idx_arr = np.asarray(idx, dtype=int)
    return X[idx_arr, :]


def _load_features_descriptors(
    df: pd.DataFrame,
    *,
    descriptor_cols: list[str] | None,
    descriptor_prefix: str | None,
    standardize: bool,
) -> tuple[np.ndarray, list[str]]:
    if descriptor_cols is None:
        if not descriptor_prefix:
            msg = "For descriptors mode, provide descriptor_cols or descriptor_prefix."
            raise ValueError(msg)
        cols = [c for c in df.columns if c.startswith(descriptor_prefix)]
    else:
        cols = list(descriptor_cols)

    if len(cols) == 0:
        msg = "No descriptor columns selected for descriptors mode."
        raise ValueError(msg)

    Xdf = df[cols].copy()
    for c in cols:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")

    if Xdf.isna().any().any():
        bad = int(Xdf.isna().sum().sum())
        msg = f"Descriptor matrix contains NaNs after coercion ({bad} NaN cells). Clean/impute first."
        raise ValueError(
            msg
        )

    X = Xdf.to_numpy(dtype=float)
    if standardize:
        X = _standardize(X)
    return X, cols


def _get_distances_for_df(df_split: pd.DataFrame, d: np.ndarray) -> np.ndarray:
    if _INTERNAL_IDX_COL not in df_split.columns:
        msg = f"Internal index column missing: {_INTERNAL_IDX_COL}"
        raise ValueError(msg)
    idx = df_split[_INTERNAL_IDX_COL].to_numpy(dtype=int)
    return d[idx]


@dataclass(frozen=True)
class DistanceAwareKFoldSplitter:
    r"""Distance-aware K-Fold splitting (OOD-oriented CV).

    This splitter creates folds by ranking samples according to their distance to
    the global centroid in feature space, then partitioning that ranking into
    disjoint test folds (farthest-first).

    Feature modes:
    - embeddings: uses an external embeddings matrix aligned by ids
    - descriptors: uses numeric columns from the input CSV

    Metrics:
    - cosine: 1 - cosine_similarity(x, centroid)
    - euclidean: ||x - centroid||

    Args:
        feature_mode: "embeddings" or "descriptors".
        metric: "cosine" or "euclidean".
        n_splits: Number of folds (>=2). Test folds are disjoint and cover the dataset.
        shuffle_ties: If True, break distance ties using a seeded tiny jitter.
        seed: Seed used for tie-breaking and optional val sampling.
        val_size: Optional validation fraction sampled from each fold's train split.
        drop_internal_index: If True, remove internal `_biosieve_row_idx__` from exported splits.

    Embeddings mode parameters:
        embeddings_path:
            Path to .npy file with shape (N, D).
        ids_path:
            CSV with embedding row ids.

    Descriptors mode parameters:
        descriptor_cols:
            Explicit descriptor columns list. If None, uses descriptor_prefix.
        descriptor_prefix:
            Select columns starting with this prefix (e.g., "desc_").
        standardize_descriptors:
            If True, z-score descriptors before distances.

    Returns:
        list[SplitResult]:
            One result per fold, each containing train/test/(optional val)
            dataframes, fold-specific params (including `fold_index`), and stats
            including distance summaries computed from the global distance vector.

    Raises:
        ValueError: If parameters are invalid, required inputs (columns/files) are missing,
        embeddings do not cover all ids, or the dataset is too small for n_splits.
        ImportError: If `val_size > 0` but scikit-learn is not installed.

    Notes:
        - This is intentionally not a random CV. It creates OOD-oriented test folds
        to evaluate robustness to distant regions of the feature space.
        - This does not prevent biological leakage by itself. Combine with group/homology/structure
        constraints if needed (future hybrid).

    Examples:
        >>> biosieve split \\
        ...   --in dataset.csv \\
        ...   --outdir runs/split_distance_aware_kfold \\
        ...   --strategy distance_aware_kfold \\
        ...   --params params.yaml

    """

    feature_mode: str = "embeddings"  # "embeddings" | "descriptors"
    metric: str = "cosine"  # "cosine" | "euclidean"

    n_splits: int = 5
    seed: int = 13
    shuffle_ties: bool = True

    val_size: float = 0.0
    drop_internal_index: bool = True

    # embeddings
    embeddings_path: str | None = None
    ids_path: str | None = None

    # descriptors
    descriptor_cols: list[str] | None = None
    descriptor_prefix: str | None = "desc_"
    standardize_descriptors: bool = True

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "distance_aware_kfold"

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> list[SplitResult]:  # noqa: C901,PLR0912,PLR0915
        """Create distance-ranked disjoint test folds with optional validation."""
        if self.n_splits < MIN_KFOLD_SPLITS:
            msg = "n_splits must be >= 2"
            raise ValueError(msg)
        if not (0.0 <= self.val_size < 1.0):
            msg = "val_size must be in [0, 1)"
            raise ValueError(msg)

        work = df.copy().reset_index(drop=True)
        work[_INTERNAL_IDX_COL] = np.arange(len(work), dtype=int)

        n = len(work)
        if n < self.n_splits:
            msg = f"Not enough samples (n={n}) for n_splits={self.n_splits}"
            raise ValueError(msg)

        feature_info: dict[str, Any] = {"feature_mode": self.feature_mode}

        # --- Load features ---
        if self.feature_mode == "embeddings":
            if not self.embeddings_path or not self.ids_path:
                msg = "Embeddings mode requires embeddings_path and ids_path."
                raise ValueError(msg)
            X = _load_features_embeddings(work, cols, self.embeddings_path, self.ids_path)
            feature_info.update(
                {
                    "embeddings_path": self.embeddings_path,
                    "ids_path": self.ids_path,
                    "n_features": int(X.shape[1]),
                }
            )

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
            msg = "feature_mode must be 'embeddings' or 'descriptors'"
            raise ValueError(msg)

        # --- Distances ---
        if self.metric == "cosine":
            d = _cosine_distance_to_centroid(X)
        elif self.metric == "euclidean":
            d = _euclidean_distance_to_centroid(X)
        else:
            msg = "metric must be 'cosine' or 'euclidean'"
            raise ValueError(msg)

        # --- Rank samples farthest -> closest ---
        rng = np.random.default_rng(self.seed)
        key = -(d + rng.normal(0.0, 1e-12, size=d.shape[0])) if self.shuffle_ties else -d
        order = np.argsort(key)

        test_slices = np.array_split(order, self.n_splits)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                msg = "val_size > 0 requires scikit-learn. Install: conda install -c conda-forge scikit-learn"
                raise ImportError(
                    msg
                )

        folds: list[SplitResult] = []
        all_idx = np.arange(n, dtype=int)

        for fold_idx, test_slice in enumerate(test_slices):
            test_idx = np.asarray(test_slice, dtype=int)

            mask = np.ones(n, dtype=bool)
            mask[test_idx] = False
            train_idx = all_idx[mask]

            train_df = work.iloc[train_idx].copy()
            test_df = work.iloc[test_idx].copy()

            # optional val sampled from fold train
            val_df: pd.DataFrame | None = None
            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)
                if tts is None:
                    msg = "val_size > 0 requires scikit-learn train_test_split."
                    raise ImportError(msg)
                train_df, val_df = tts(
                    train_df,
                    test_size=self.val_size,
                    random_state=seed_fold,
                    shuffle=True,
                    stratify=None,
                )

            # distance stats using internal indices (exact for train/test/val)
            d_train = _get_distances_for_df(train_df, d)
            d_test = _get_distances_for_df(test_df, d)
            d_val = _get_distances_for_df(val_df, d) if val_df is not None else None

            stats: dict[str, Any] = {
                "fold_index": int(fold_idx),
                "n_total": len(df),
                "n_used": int(n),
                "n_train": len(train_df),
                "n_test": len(test_df),
                "n_val": len(val_df) if val_df is not None else 0,
                "metric": self.metric,
                "distance_stats_global": _label_stats_from_array(d),
                "distance_stats_train": _label_stats_from_array(d_train),
                "distance_stats_test": _label_stats_from_array(d_test),
                "distance_stats_val": _label_stats_from_array(d_val) if d_val is not None else None,
            }

            # clean export frames (optional)
            if self.drop_internal_index:
                train_df = train_df.drop(columns=[_INTERNAL_IDX_COL]).reset_index(drop=True)
                test_df = test_df.drop(columns=[_INTERNAL_IDX_COL]).reset_index(drop=True)
                if val_df is not None:
                    val_df = val_df.drop(columns=[_INTERNAL_IDX_COL]).reset_index(drop=True)
            else:
                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)
                if val_df is not None:
                    val_df = val_df.reset_index(drop=True)

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
                        "drop_internal_index": self.drop_internal_index,
                        **feature_info,
                        "fold_index": int(fold_idx),
                    },
                    stats=stats,
                )
            )

        return folds
