from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from biosieve.types import Columns
from biosieve.splitting.base import SplitResult


def _try_import_group_kfold():
    try:
        from sklearn.model_selection import GroupKFold  # type: ignore
        return GroupKFold
    except Exception:
        return None


def _try_import_train_test_split():
    try:
        from sklearn.model_selection import train_test_split  # type: ignore
        return train_test_split
    except Exception:
        return None


@dataclass(frozen=True)
class GroupKFoldSplitter:
    """
    Group K-Fold splitting (leakage-aware).

    Ensures that the same group does not appear in both train and test for any fold.

    Produces a list of SplitResult objects, one per fold:
      - train: training subset for that fold
      - test:  held-out subset for that fold
      - val:   optional validation subset sampled from train (val_size > 0)

    Parameters
    ----------
    group_col:
        Column containing group identifiers (e.g., subject_id, cluster_id, taxid).
    n_splits:
        Number of folds.
    seed:
        Used only for the optional val split inside the train fold.
    val_size:
        Optional validation fraction sampled from the fold's train set.

    Notes
    -----
    - `GroupKFold` does not support shuffle; folds are deterministic given group ordering.
      If you want shuffled group folds, we can add `GroupShuffleSplit`-based CV later.
    """

    group_col: str = "group"
    n_splits: int = 5

    # Optional val inside train for each fold
    val_size: float = 0.0
    seed: int = 13

    dropna: bool = True  # drop rows with NaN group ids

    @property
    def strategy(self) -> str:
        return "group_kfold"

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> List[SplitResult]:
        GroupKFold = _try_import_group_kfold()
        if GroupKFold is None:
            raise ImportError(
                "GroupKFoldSplitter requires scikit-learn. "
                "Install: conda install -c conda-forge scikit-learn"
            )

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.val_size < 0 or self.val_size >= 1:
            raise ValueError("val_size must be in [0, 1)")
        if self.group_col not in df.columns:
            raise ValueError(
                f"Missing group column '{self.group_col}'. Columns: {df.columns.tolist()}"
            )

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
                raise ValueError(
                    f"Found NaN groups in '{self.group_col}'. Set dropna=true or clean dataset."
                )
            dropped = 0
            g = g.astype(str)

        n_groups = int(pd.Series(g).nunique(dropna=False))
        if n_groups < self.n_splits:
            raise ValueError(
                f"Not enough unique groups for n_splits={self.n_splits}. "
                f"Found n_groups={n_groups}. Need at least n_splits groups."
            )

        gkf = GroupKFold(n_splits=self.n_splits)

        tts = None
        if self.val_size and self.val_size > 0:
            tts = _try_import_train_test_split()
            if tts is None:
                raise ImportError("val_size > 0 requires scikit-learn train_test_split.")

        folds: List[SplitResult] = []

        # X can be dummy; GroupKFold uses indices + groups only
        X_dummy = work.index.values

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X_dummy, y=None, groups=g)):
            train_df = work.iloc[train_idx].copy().reset_index(drop=True)
            test_df = work.iloc[test_idx].copy().reset_index(drop=True)

            # leakage check (group disjointness)
            train_groups = set(train_df[self.group_col].astype(str).unique())
            test_groups = set(test_df[self.group_col].astype(str).unique())
            leak = len(train_groups & test_groups)

            val_df: Optional[pd.DataFrame] = None
            val_leak_train = 0
            val_leak_test = 0

            if self.val_size and self.val_size > 0:
                seed_fold = int(self.seed + fold_idx)
                train_df, val_df = tts(
                    train_df,
                    test_size=self.val_size,
                    random_state=seed_fold,
                    shuffle=True,
                    stratify=None,
                )
                train_df = train_df.reset_index(drop=True)
                val_df = val_df.reset_index(drop=True)

                val_groups = set(val_df[self.group_col].astype(str).unique())
                train_groups2 = set(train_df[self.group_col].astype(str).unique())
                val_leak_train = len(val_groups & train_groups2)
                val_leak_test = len(val_groups & test_groups)

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
                        "fold_index": fold_idx,
                    },
                    stats={
                        "fold_index": int(fold_idx),
                        "n_total": int(len(df)),
                        "n_used": int(len(work)),
                        "n_dropped_nan": int(dropped),
                        "n_train": int(len(train_df)),
                        "n_test": int(len(test_df)),
                        "n_val": int(len(val_df)) if val_df is not None else 0,
                        "n_groups_total": int(n_groups),
                        "n_groups_train": int(len(train_groups)),
                        "n_groups_test": int(len(test_groups)),
                        "leak_groups_train_test": int(leak),
                        "leak_groups_val_train": int(val_leak_train),
                        "leak_groups_val_test": int(val_leak_test),
                    },
                )
            )

        return folds
