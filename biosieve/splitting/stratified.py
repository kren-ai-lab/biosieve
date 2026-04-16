"""Stratified splitting strategy for classification labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


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


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        return cast("_TrainTestSplitFn", train_test_split)
    except ImportError:
        return None


def _validate_sizes(test_size: float, val_size: float) -> None:
    if not (0.0 < test_size < 1.0):
        msg = "test_size must be in (0, 1)"
        raise ValueError(msg)
    if not (0.0 <= val_size < 1.0):
        msg = "val_size must be in [0, 1)"
        raise ValueError(msg)
    if test_size + val_size >= 1.0:
        msg = "test_size + val_size must be < 1.0"
        raise ValueError(msg)


def _validate_inputs(
    df: pl.DataFrame,
    label_col: str,
    test_size: float,
    val_size: float,
    *,
    dropna: bool,
) -> pl.DataFrame:
    _validate_sizes(test_size, val_size)
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


def _value_counts_dict(series: pl.Series) -> dict[str, int]:
    return {
        str(row[0]): int(row[1])
        for row in series.cast(pl.String).value_counts().iter_rows()
    }


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

        tts = _try_import_train_test_split()
        if tts is None:
            msg = (
                "StratifiedSplitter requires scikit-learn. Install: conda install -c conda-forge scikit-learn"
            )
            raise ImportError(msg)

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
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                msg = "Derived val fraction invalid. Check test_size/val_size."
                raise ValueError(msg)

            y_tv = trainval[self.label_col].to_numpy()
            inner_idx = np.arange(trainval.height)
            train_idx, val_idx = tts(
                inner_idx,
                test_size=frac,
                random_state=self.seed,
                shuffle=True,
                stratify=y_tv,
            )
            train = trainval[train_idx]
            val = trainval[val_idx]

        stats: dict[str, Any] = {
            "n_total": work.height,
            "n_train": train.height,
            "n_test": test.height,
            "n_val": val.height if val is not None else 0,
            "label_col": self.label_col,
            "seed": int(self.seed),
            "test_size": float(self.test_size),
            "val_size": float(self.val_size),
            "train_label_counts": _value_counts_dict(train[self.label_col]),
            "test_label_counts": _value_counts_dict(test[self.label_col]),
        }

        log.info(
            "stratified:stats | train=%d | val=%d | test=%d",
            train.height,
            int(val.height if val is not None else 0),
            test.height,
        )

        if val is not None:
            stats["val_label_counts"] = _value_counts_dict(val[self.label_col])

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
