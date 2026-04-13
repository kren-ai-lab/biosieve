"""Group-aware k-fold splitter for leakage-safe cross-validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from biosieve.types import Columns

log = get_logger(__name__)
MIN_KFOLD_SPLITS = 2


class _GroupKFold(Protocol):
    def split(self, X: object, y: object, groups: object) -> Iterator[tuple[list[int], list[int]]]: ...


class _GroupKFoldFactory(Protocol):
    def __call__(self, *, n_splits: int) -> _GroupKFold: ...


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


def _try_import_group_kfold() -> _GroupKFoldFactory | None:
    try:
        from sklearn.model_selection import GroupKFold  # noqa: PLC0415

        return cast("_GroupKFoldFactory", GroupKFold)
    except ImportError:
        return None


def _try_import_train_test_split() -> _TrainTestSplitFn | None:
    try:
        from sklearn.model_selection import train_test_split  # noqa: PLC0415

        return cast("_TrainTestSplitFn", train_test_split)
    except ImportError:
        return None


def _group_set(df: pd.DataFrame, group_col: str) -> set[str]:
    return set(df[group_col].astype(str).unique())


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

    def run_folds(self, df: pd.DataFrame, _cols: Columns) -> list[SplitResult]:  # noqa: C901,PLR0912,PLR0915
        """Build group-disjoint folds with optional per-fold validation splits."""
        GroupKFold = _try_import_group_kfold()
        if GroupKFold is None:
            msg = (
                "GroupKFoldSplitter requires scikit-learn. Install: conda install -c conda-forge scikit-learn"
            )
            raise ImportError(msg)

        if self.n_splits < MIN_KFOLD_SPLITS:
            msg = "n_splits must be >= 2"
            raise ValueError(msg)
        if not (0.0 <= self.val_size < 1.0):
            msg = "val_size must be in [0, 1)"
            raise ValueError(msg)
        if self.group_col not in df.columns:
            msg = f"Missing group column '{self.group_col}'. Columns: {df.columns.tolist()}"
            raise ValueError(msg)

        work = df.copy().reset_index(drop=True)
        g = work[self.group_col]

        # handle missing groups
        if self.dropna:
            keep = ~g.isna()
            dropped = int((~keep).sum())
            work = work.loc[keep].reset_index(drop=True)
            g = work[self.group_col].astype(str).reset_index(drop=True)
        else:
            if g.isna().any():
                msg = f"Found NaN groups in '{self.group_col}'. Set dropna=true or clean dataset."
                raise ValueError(msg)
            dropped = 0
            g = g.astype(str)

        n_groups = int(pd.Series(g).nunique(dropna=False))
        if n_groups < self.n_splits:
            msg = (
                f"Not enough unique groups for n_splits={self.n_splits}. "
                f"Found n_groups={n_groups}. Need at least n_splits groups."
            )
            raise ValueError(msg)

        gkf = GroupKFold(n_splits=self.n_splits)

        tts = None
        if self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                msg = "val_size > 0 requires scikit-learn train_test_split."
                raise ImportError(msg)

        folds: list[SplitResult] = []

        # X can be dummy; GroupKFold uses indices + groups only
        X_dummy = work.index.to_numpy()

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y=None, groups=g)):
            train_df = work.iloc[train_idx].copy().reset_index(drop=True)
            test_df = work.iloc[test_idx].copy().reset_index(drop=True)

            # leakage check (group disjointness)
            train_groups = _group_set(train_df, self.group_col)
            test_groups = _group_set(test_df, self.group_col)
            leak_tt = len(train_groups & test_groups)

            if leak_tt != 0:
                msg = (
                    f"Group leakage detected in fold {fold_idx}: "
                    f"train/test share {leak_tt} group(s). This should never happen."
                )
                raise ValueError(msg)

            val_df: pd.DataFrame | None = None
            leak_vt = 0
            leak_vr = 0

            if self.val_size > 0:
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
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)

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
