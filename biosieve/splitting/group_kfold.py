"""Group-aware k-fold splitter for leakage-safe cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from biosieve.splitting.base import SplitResult
from biosieve.splitting.common import (
    sklearn_required_message,
    split_train_val,
    validate_kfold,
)
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from biosieve.types import Columns

log = get_logger(__name__)


def _group_set(df: pl.DataFrame, group_col: str) -> set[str]:
    return set(df[group_col].cast(pl.String).unique().to_list())


@dataclass(frozen=True)
class GroupKFoldSplitter:
    r"""Group K-Fold splitting (leakage-aware cross-validation).

    Ensures that the same group does not appear in both train and test for any fold.

    Output:
    Produces a list of SplitResult objects, one per fold:
      - train: training subset for that fold
      - test:  held-out subset for that fold (group-disjoint from train)
      - val:   optional validation subset sampled from train (val_size > 0)

    Args:
        group_col: Column containing group identifiers (e.g., subject_id, cluster_id, taxid).
        n_splits: Number of folds (must be >= 2).
        val_size: Optional validation fraction sampled from each fold's *train* split.
        Set to 0.0 to disable validation.
        seed: Seed used only for the optional val split inside each fold.
        dropna: If True, drop rows with NaN group ids. If False, raise when NaNs are present.

    Returns:
        One SplitResult per fold. Each SplitResult includes:
        - params: effective fold parameters (includes fold_index)
        - stats: counts and leakage checks:
        - leak_groups_train_test must be 0: - leak_groups_val_test should be 0 (val sampled from train)

    Raises:
        ImportError: If scikit-learn is not installed.
        ValueError: If required columns are missing, parameter ranges are invalid, NaNs are present
        and dropna=False, or there are insufficient unique groups for n_splits.

    Notes:
        - `GroupKFold` does not support shuffling; folds are deterministic given the group
        ordering in the input. If you require shuffled group CV, consider a future
        `group_shuffle_kfold` strategy based on `GroupShuffleSplit`.
        - Validation is sampled from the training fold. It may share groups with train
        (by design), but should never include groups from the test fold.

    Examples:
        >>> biosieve split \\
        ...   --in dataset.csv \\
        ...   --outdir runs/split_group_kfold \\
        ...   --strategy group_kfold \\
        ...   --params params.yaml

    """

    group_col: str = "group"
    n_splits: int = 5

    # Optional val inside train for each fold
    val_size: float = 0.0
    seed: int = 13

    dropna: bool = True

    @property
    def strategy(self) -> str:
        """Return the strategy identifier."""
        return "group_kfold"

    def run_folds(self, df: pl.DataFrame, _cols: Columns) -> list[SplitResult]:
        """Build group-disjoint folds with optional per-fold validation splits."""
        try:
            from sklearn.model_selection import GroupKFold  # noqa: PLC0415
        except ImportError as e:
            msg = sklearn_required_message("GroupKFoldSplitter")
            raise ImportError(msg) from e

        if self.group_col not in df.columns:
            msg = f"Missing group column '{self.group_col}'. Columns: {df.columns}"
            raise ValueError(msg)

        work = df.clone()
        g = work[self.group_col]

        # handle missing groups
        if self.dropna:
            keep = g.is_not_null()
            dropped = int((~keep).sum())
            work = work.filter(keep)
            g = work[self.group_col].cast(pl.String)
        else:
            if g.is_null().any():
                msg = f"Found NaN groups in '{self.group_col}'. Set dropna=true or clean dataset."
                raise ValueError(msg)
            dropped = 0
            g = g.cast(pl.String)

        n_groups = int(g.n_unique())
        validate_kfold(self.n_splits, self.val_size)
        if n_groups < self.n_splits:
            msg = (
                f"Not enough unique groups for n_splits={self.n_splits}. "
                f"Found n_groups={n_groups}. Need at least n_splits groups."
            )
            raise ValueError(msg)

        gkf = GroupKFold(n_splits=self.n_splits)

        folds: list[SplitResult] = []

        # X can be dummy; GroupKFold uses indices + groups only
        X_dummy = np.arange(work.height)

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y=None, groups=g.to_numpy())):
            train_df = work[train_idx]
            test_df = work[test_idx]

            train_groups = _group_set(train_df, self.group_col)
            test_groups = _group_set(test_df, self.group_col)
            leak_tt = len(train_groups & test_groups)

            val_df: pl.DataFrame | None = None
            leak_vt = 0
            leak_vr = 0

            if self.val_size > 0:
                train_df, val_df = split_train_val(
                    train_df,
                    val_size=self.val_size,
                    seed=int(self.seed + fold_idx),
                    feature="GroupKFoldSplitter with val_size > 0",
                )

                val_groups = _group_set(val_df, self.group_col)
                train_groups2 = _group_set(train_df, self.group_col)
                leak_vr = len(val_groups & train_groups2)  # expected possibly >0
                leak_vt = len(val_groups & test_groups)  # should be 0

            folds.append(
                SplitResult(
                    train=train_df,
                    test=test_df,
                    val=val_df,
                    strategy=self.strategy,
                    params={
                        "group_col": self.group_col,
                        "n_splits": self.n_splits,
                        "val_size": self.val_size,
                        "seed": self.seed,
                        "dropna": self.dropna,
                        "fold_index": int(fold_idx),
                    },
                    stats={
                        "fold_index": int(fold_idx),
                        "n_total": len(df),
                        "n_used": len(work),
                        "n_dropped_nan": int(dropped),
                        "n_train": len(train_df),
                        "n_test": len(test_df),
                        "n_val": len(val_df) if val_df is not None else 0,
                        "n_groups_total": int(n_groups),
                        "n_groups_train": len(train_groups),
                        "n_groups_test": len(test_groups),
                        "leak_groups_train_test": int(leak_tt),  # must be 0
                        "leak_groups_val_train": int(leak_vr),  # may be >0 (by design)
                        "leak_groups_val_test": int(leak_vt),  # should be 0
                    },
                )
            )

        return folds
