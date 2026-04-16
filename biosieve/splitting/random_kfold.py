"""Random k-fold splitting baseline strategy."""

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


class _KFold(Protocol):
    def split(self, X: object) -> Iterator[tuple[list[int], list[int]]]: ...


class _KFoldFactory(Protocol):
    def __call__(self, *, n_splits: int, shuffle: bool, random_state: int) -> _KFold: ...


class _TrainTestSplitFn(Protocol):
    def __call__(
        self,
        X: object,
        *,
        test_size: float,
        random_state: int,
        shuffle: bool,
        stratify: None,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _try_import_kfold() -> _KFoldFactory | None:
    try:
        from sklearn.model_selection import KFold  # noqa: PLC0415

        return cast("_KFoldFactory", KFold)
    except ImportError:
        return None


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        return cast("_TrainTestSplitFn", train_test_split)
    except ImportError:
        return None


@dataclass(frozen=True)
class RandomKFoldSplitter:
    """Random K-Fold splitting.

    Produces a list of SplitResult objects, one per fold:
      - train: training subset for that fold
      - test:  held-out subset for that fold
      - val:   optional validation subset sampled from train (val_size > 0)

    Notes:
        - This is *random* KFold, not stratified.
        - For reproducibility, each fold uses seed + fold_index for the optional val split.

    """

    n_splits: int = 5
    shuffle: bool = True
    seed: int = 13

    # optional val inside train for each fold
    val_size: float = 0.0

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "random_kfold"

    def run_folds(self, df: pl.DataFrame, _cols: Columns) -> list[SplitResult]:
        """Create random k-fold splits with optional per-fold validation."""
        KFold = _try_import_kfold()
        if KFold is None:
            msg = (
                "RandomKFoldSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )
            raise ImportError(msg)

        if self.n_splits < MIN_KFOLD_SPLITS:
            msg = "n_splits must be >= 2"
            raise ValueError(msg)
        if self.val_size < 0 or self.val_size >= 1:
            msg = "val_size must be in [0, 1)"
            raise ValueError(msg)

        work = df.clone()
        n = work.height
        if n < self.n_splits:
            msg = f"Not enough samples (n={n}) for n_splits={self.n_splits}"
            raise ValueError(msg)

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                msg = "val_size > 0 requires scikit-learn train_test_split. Install scikit-learn."
                raise ImportError(msg)

        folds: list[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
            train_df = work[train_idx]
            test_df = work[test_idx]

            val_df: pl.DataFrame | None = None

            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)
                if tts is None:
                    msg = "val_size > 0 requires scikit-learn train_test_split. Install scikit-learn."
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
                        "n_splits": self.n_splits,
                        "shuffle": self.shuffle,
                        "seed": self.seed,
                        "val_size": self.val_size,
                        "fold_index": fold_idx,
                    },
                    stats={
                        "fold_index": int(fold_idx),
                        "n_total": int(n),
                        "n_train": train_df.height,
                        "n_test": test_df.height,
                        "n_val": val_df.height if val_df is not None else 0,
                    },
                )
            )

        return folds
