"""Tests for GroupKFoldSplitter — includes leakage invariant per fold."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pytest


import pytest

from biosieve.splitting.group_kfold import GroupKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
# 5 groups → 3 splits is valid (GroupKFold requires n_splits <= n_groups)
N_SPLITS = 3


def test_leakage_zero_per_fold(df_grouped: pd.DataFrame) -> None:
    """Core invariant: no group appears in both train and test within any fold."""
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13)
    folds = splitter.run_folds(df_grouped, COLS)
    for res in folds:
        assert res.stats["leak_groups_train_test"] == 0
        train_groups = set(res.train["group"])
        test_groups = set(res.test["group"])
        assert train_groups & test_groups == set()


def test_missing_group_col_raises(df_base: pd.DataFrame) -> None:
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="NONEXISTENT")
    with pytest.raises((ValueError, KeyError)):
        splitter.run_folds(df_base, COLS)
