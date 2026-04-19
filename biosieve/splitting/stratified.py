"""Stratified splitting strategy for classification labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import (
    derive_val_fraction,
    require_train_test_split,
    split_train_val,
    validate_sizes,
    value_counts_dict,
)
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    import polars as pl

    from biosieve.types import Columns

log = get_logger(__name__)


def _validate_inputs(
    df: pl.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    *,
    dropna: bool,
) -> pl.DataFrame:
    validate_sizes(test_size, val_size)
    if label_col not in df.columns:
        msg = f"Missing label column '{label_col}'. Columns: {df.columns}"
        raise ValueError(msg)

    y = df[label_col]
    if y.is_null().any():
        if not dropna:
            msg = f"Found NaN labels in '{label_col}'. Set dropna=true or clean dataset."
            raise ValueError(msg)
        keep = y.is_not_null()
        return df.filter(keep)
    return df


@dataclass(frozen=True)
class StratifiedSplitter:
    r"""Stratified train/test(/val) split for classification.

    This strategy preserves class proportions in the test set (and optionally the
    validation set) using scikit-learn's `train_test_split(..., stratify=y)`.

    Args:
        label_col: Column containing class labels.
        test_size: Fraction of samples assigned to the test set.
        val_size: Fraction of samples assigned to the validation set (0 disables validation).
            This fraction is relative to the full dataset and is internally converted
            to a fraction of the remaining train+val set.
        seed: Random seed used by scikit-learn.
        dropna: If True, drop rows with NaN labels. If False, raise on NaNs.

    Returns:
        A container with:
        - train/test/val DataFrames: - params: {"label_col","test_size","val_size","seed","dropna"}
        - stats: counts and class distributions per split

    Raises:
        ImportError: If scikit-learn is not available.
        ValueError: If label column is missing, sizes invalid, NaNs present (dropna=False),
        or stratification cannot be performed (e.g., too few samples in a class).

    Notes:
        - Use this for classification tasks when you do not need group/leakage constraints.
        If you have groups/clusters/homology, prefer `group`/`group_kfold` or hybrids.
        - Stratification requires that each class has enough members to be split; otherwise
        scikit-learn raises a ValueError.

    Examples:
        >>> biosieve split \\
        ...   --in dataset.csv \\
        ...   --outdir runs/split_stratified \\
        ...   --strategy stratified \\
        ...   --params params.yaml

    """

    label_col: str = "label"
    test_size: float = 0.2
    val_size: float = 0.0
    seed: int = 13
    dropna: bool = True

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "stratified"

    def run(self, df: pl.DataFrame, cols: Columns) -> SplitResult:
        """Create stratified train/test/(val) partitions."""
        log.info(
            "stratified:start | label_col=%s | test_size=%.3f | val_size=%.3f",
            cols.label_col,
            self.test_size,
            self.val_size,
        )
        log.debug("stratified:params | %s", self.__dict__)

        tts = require_train_test_split("StratifiedSplitter")

        work = _validate_inputs(
            df.clone(),
            self.label_col,
            self.test_size,
            self.val_size,
            dropna=self.dropna,
        )
        y = work[self.label_col].to_numpy()

        # 1) split off test
        all_idx = np.arange(work.height)
        trainval_idx, test_idx = tts(
            all_idx,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=y,
        )
        trainval = work[trainval_idx]
        test = work[test_idx]

        val = None
        train = trainval

        # 2) optional split train/val (stratified within trainval)
        if self.val_size and self.val_size > 0:
            frac = derive_val_fraction(self.test_size, self.val_size)
            y_tv = trainval[self.label_col].to_numpy()
            train, val = split_train_val(
                trainval,
                val_size=frac,
                seed=self.seed,
                feature="StratifiedSplitter",
                stratify=y_tv,
                train_test_split=tts,
            )

        stats: dict[str, Any] = {
            "n_total": work.height,
            "n_train": train.height,
            "n_test": test.height,
            "n_val": val.height if val is not None else 0,
            "label_col": self.label_col,
            "seed": int(self.seed),
            "test_size": float(self.test_size),
            "val_size": float(self.val_size),
            "train_label_counts": value_counts_dict(train[self.label_col]),
            "test_label_counts": value_counts_dict(test[self.label_col]),
        }

        log.info(
            "stratified:stats | train=%d | val=%d | test=%d",
            train.height,
            int(val.height if val is not None else 0),
            test.height,
        )

        if val is not None:
            stats["val_label_counts"] = value_counts_dict(val[self.label_col])

        return SplitResult(
            train=train,
            test=test,
            val=val,
            strategy=self.strategy,
            params={
                "label_col": self.label_col,
                "test_size": self.test_size,
                "val_size": self.val_size,
                "seed": self.seed,
                "dropna": self.dropna,
            },
            stats=stats,
        )
