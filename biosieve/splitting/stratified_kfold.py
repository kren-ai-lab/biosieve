"""Stratified k-fold splitting strategy for classification tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import (
    sklearn_required_message,
    split_train_val,
    validate_kfold,
    value_counts_dict,
)
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)
MIN_KFOLD_SPLITS = 2


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

    def run_folds(self, df: pl.DataFrame, _cols: Columns) -> list[SplitResult]:
        """Create stratified k-fold splits with optional per-fold validation."""
        try:
            from sklearn.model_selection import StratifiedKFold  # noqa: PLC0415
        except ImportError as e:
            msg = sklearn_required_message("StratifiedKFoldSplitter")
            raise ImportError(msg) from e

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
        validate_kfold(self.n_splits, self.val_size, n_samples=n)

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

        folds: list[SplitResult] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.arange(n), y.to_numpy())):
            train_df = work[train_idx]
            test_df = work[test_idx]

            val_df: pl.DataFrame | None = None

            if self.val_size and self.val_size > 0:
                train_df, val_df = split_train_val(
                    train_df,
                    val_size=self.val_size,
                    seed=int(self.seed + fold_idx),
                    feature="StratifiedKFoldSplitter with val_size > 0",
                )

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
                        "train_label_counts": value_counts_dict(cast("Any", train_df[self.label_col])),
                        "test_label_counts": value_counts_dict(cast("Any", test_df[self.label_col])),
                    },
                )
            )

        return folds
