"""Stratified k-fold splitting strategy for classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from biosieve.types import Columns

log = get_logger(__name__)
MIN_KFOLD_SPLITS = 2


class _StratifiedKFold(Protocol):
    def split(self, X: object, y: object) -> Iterator[tuple[list[int], list[int]]]: ...


class _StratifiedKFoldFactory(Protocol):
    def __call__(self, *, n_splits: int, shuffle: bool, random_state: int) -> _StratifiedKFold: ...


class _TrainTestSplitFn(Protocol):
    def __call__(
        self,
        X: object,
        *,
        test_size: float,
        random_state: int,
        shuffle: bool,
        stratify: object,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _try_import_stratified_kfold() -> _StratifiedKFoldFactory | None:
    try:
        from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415

        return cast("_StratifiedKFoldFactory", StratifiedKFold)
    except ImportError:
        return None


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        return cast("_TrainTestSplitFn", train_test_split)
    except ImportError:
        return None


@dataclass(frozen=True)
class StratifiedKFoldSplitter:
    """Stratified K-Fold splitting for classification labels.

    Produces a list of SplitResult objects, one per fold:
      - train: training subset for that fold
      - test:  held-out subset for that fold
      - val:   optional validation subset sampled from train (val_size > 0)

    Notes:
        - Stratification is performed using `label_col`.
        - If val_size > 0, validation is sampled randomly from the fold's train
        (non-stratified by default to keep it simple and robust).
        If you want stratified val too, we can add `val_stratify=true`.

    """

    label_col: str = "label"

    n_splits: int = 5
    shuffle: bool = True
    seed: int = 13

    # optional validation inside train for each fold
    val_size: float = 0.0

    # behavior
    dropna: bool = True  # drop rows with NaN label
    cast_to_str: bool = False  # optionally cast labels to str (useful if labels are mixed types)

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "stratified_kfold"

    def run_folds(self, df: pl.DataFrame, _cols: Columns) -> list[SplitResult]:  # noqa: C901,PLR0912,PLR0915
        """Create stratified k-fold splits with optional per-fold validation."""
        StratifiedKFold = _try_import_stratified_kfold()
        if StratifiedKFold is None:
            msg = (
                "StratifiedKFoldSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )
            raise ImportError(msg)

        if self.n_splits < MIN_KFOLD_SPLITS:
            msg = "n_splits must be >= 2"
            raise ValueError(msg)
        if self.val_size < 0 or self.val_size >= 1:
            msg = "val_size must be in [0, 1)"
            raise ValueError(msg)
        if self.label_col not in df.columns:
            msg = f"Missing label column '{self.label_col}'. Columns: {df.columns}"
            raise ValueError(msg)

        work = df.clone()

        y = work[self.label_col]

        # handle missing labels
        if self.dropna:
            keep = y.is_not_null()
            dropped = int((~keep).sum())
            work = work.filter(keep)
            y = work[self.label_col]
        else:
            if y.is_null().any():
                msg = f"Found NaN labels in '{self.label_col}'. Set dropna=true or clean dataset."
                raise ValueError(msg)
            dropped = 0

        if self.cast_to_str:
            y = y.cast(pl.String)

        n = work.height
        if n < self.n_splits:
            msg = f"Not enough samples (n={n}) for n_splits={self.n_splits}"
            raise ValueError(msg)

        # sanity: each class must have at least n_splits members for StratifiedKFold
        vc = {str(row[0]): int(row[1]) for row in y.cast(pl.String).value_counts().iter_rows()}
        too_small = {k: v for k, v in vc.items() if v < self.n_splits}
        if too_small:
            msg = (
                "Some classes have fewer samples than n_splits, cannot stratify k-fold. "
                f"n_splits={self.n_splits}. Problem classes: {too_small}"
            )
            raise ValueError(msg)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                msg = "val_size > 0 requires scikit-learn train_test_split."
                raise ImportError(msg)

        folds: list[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(n), y.to_numpy())):
            train_df = work[train_idx]
            test_df = work[test_idx]

            val_df: pl.DataFrame | None = None

            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)

                # default: random val from train (robust)
                if tts is None:
                    msg = "val_size > 0 requires scikit-learn train_test_split."
                    raise ImportError(msg)
                inner_idx = np.arange(train_df.height)
                train_keep_idx, val_idx = tts(
                    inner_idx,
                    test_size=self.val_size,
                    random_state=seed_fold,
                    shuffle=True,
                    stratify=None,
                )
                val_df = train_df[val_idx]
                train_df = train_df[train_keep_idx]

            folds.append(
                SplitResult(
                    train=train_df,
                    test=test_df,
                    val=val_df,
                    strategy=self.strategy,
                    params={
                        "label_col": self.label_col,
                        "n_splits": self.n_splits,
                        "shuffle": self.shuffle,
                        "seed": self.seed,
                        "val_size": self.val_size,
                        "dropna": self.dropna,
                        "cast_to_str": self.cast_to_str,
                        "fold_index": fold_idx,
                    },
                    stats={
                        "fold_index": int(fold_idx),
                        "n_total": df.height,
                        "n_used": int(n),
                        "n_dropped_nan": int(dropped),
                        "n_train": train_df.height,
                        "n_test": test_df.height,
                        "n_val": val_df.height if val_df is not None else 0,
                        "train_label_counts": {
                            str(row[0]): int(row[1])
                            for row in train_df[self.label_col].cast(pl.String).value_counts().iter_rows()
                        },
                        "test_label_counts": {
                            str(row[0]): int(row[1])
                            for row in test_df[self.label_col].cast(pl.String).value_counts().iter_rows()
                        },
                    },
                )
            )

        return folds
