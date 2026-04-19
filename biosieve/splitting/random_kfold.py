"""Random k-fold splitting baseline strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import (
    sklearn_required_message,
    split_train_val,
    validate_kfold,
)
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    import polars as pl

    from biosieve.types import Columns

log = get_logger(__name__)
MIN_KFOLD_SPLITS = 2


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
        try:
            from sklearn.model_selection import KFold  # noqa: PLC0415
        except ImportError as e:
            msg = sklearn_required_message("RandomKFoldSplitter")
            raise ImportError(msg) from e

        work = df.clone()
        n = work.height
        validate_kfold(self.n_splits, self.val_size, n_samples=n)

        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.seed)

        folds: list[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(np.arange(n))):
            train_df = work[train_idx]
            test_df = work[test_idx]

            val_df: pl.DataFrame | None = None

            if self.val_size and self.val_size > 0:
                train_df, val_df = split_train_val(
                    train_df,
                    val_size=self.val_size,
                    seed=int(self.seed + fold_idx),
                    feature="RandomKFoldSplitter with val_size > 0",
                )

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
