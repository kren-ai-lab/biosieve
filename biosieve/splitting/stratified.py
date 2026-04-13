from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

    from biosieve.types import Columns

log = get_logger(__name__)


class _TrainTestSplitFn(Protocol):
    def __call__(
        self,
        df: pd.DataFrame,
        *,
        test_size: float,
        random_state: int,
        shuffle: bool,
        stratify: pd.Series | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]: ...


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split

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


@dataclass(frozen=True)
class StratifiedSplitter:
    """Stratified train/test(/val) split for classification.

    This strategy preserves class proportions in the test set (and optionally the
    validation set) using scikit-learn's `train_test_split(..., stratify=y)`.

    Parameters
    ----------
    label_col:
        Column containing class labels.
    test_size:
        Fraction of samples assigned to the test set.
    val_size:
        Fraction of samples assigned to the validation set (0 disables validation).
        This fraction is relative to the full dataset and is internally converted
        to a fraction of the remaining train+val set.
    seed:
        Random seed used by scikit-learn.
    dropna:
        If True, drop rows with NaN labels. If False, raise on NaNs.

    Returns
    -------
    SplitResult
        A container with:
        - train/test/val DataFrames
        - params: {"label_col","test_size","val_size","seed","dropna"}
        - stats: counts and class distributions per split

    Raises
    ------
    ImportError
        If scikit-learn is not available.
    ValueError
        If label column is missing, sizes invalid, NaNs present (dropna=False),
        or stratification cannot be performed (e.g., too few samples in a class).

    Notes
    -----
    - Use this for classification tasks when you do not need group/leakage constraints.
      If you have groups/clusters/homology, prefer `group`/`group_kfold` or hybrids.
    - Stratification requires that each class has enough members to be split; otherwise
      scikit-learn raises a ValueError.

    Examples
    --------
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
        return "stratified"

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:

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
                "StratifiedSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )
            raise ImportError(
                msg
            )

        _validate_sizes(self.test_size, self.val_size)

        work = df.copy().reset_index(drop=True)
        if self.label_col not in work.columns:
            msg = f"Missing label column '{self.label_col}'. Columns: {work.columns.tolist()}"
            raise ValueError(msg)

        y = work[self.label_col]
        if y.isna().any():
            if self.dropna:
                keep = ~y.isna()
                work = work.loc[keep].reset_index(drop=True)
                y = work[self.label_col].reset_index(drop=True)
            else:
                msg = f"Found NaN labels in '{self.label_col}'. Set dropna=true or clean dataset."
                raise ValueError(msg)

        # 1) split off test
        trainval, test = tts(
            work,
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=True,
            stratify=y,
        )

        val = None
        train = trainval

        # 2) optional split train/val (stratified within trainval)
        if self.val_size and self.val_size > 0:
            frac = self.val_size / (1.0 - self.test_size)
            if frac <= 0 or frac >= 1:
                msg = "Derived val fraction invalid. Check test_size/val_size."
                raise ValueError(msg)

            y_tv = trainval[self.label_col]
            train, val = tts(
                trainval,
                test_size=frac,
                random_state=self.seed,
                shuffle=True,
                stratify=y_tv,
            )

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        if val is not None:
            val = val.reset_index(drop=True)

        stats: dict[str, Any] = {
            "n_total": len(work),
            "n_train": len(train),
            "n_test": len(test),
            "n_val": len(val) if val is not None else 0,
            "label_col": self.label_col,
            "seed": int(self.seed),
            "test_size": float(self.test_size),
            "val_size": float(self.val_size),
            "train_label_counts": train[self.label_col].astype(str).value_counts(dropna=False).to_dict(),
            "test_label_counts": test[self.label_col].astype(str).value_counts(dropna=False).to_dict(),
        }

        log.info(
            "stratified:stats | train=%d | val=%d | test=%d",
            len(train),
            int(len(val) if val is not None else 0),
            len(test),
        )

        if val is not None:
            stats["val_label_counts"] = val[self.label_col].astype(str).value_counts(dropna=False).to_dict()

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
