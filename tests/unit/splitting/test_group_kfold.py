"""Tests for GroupKFoldSplitter — includes leakage invariant per fold."""

from __future__ import annotations

import pytest

from biosieve.splitting.group_kfold import GroupKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
# 5 groups → 3 splits is valid (GroupKFold requires n_splits <= n_groups)
N_SPLITS = 3


def test_returns_n_folds(df_grouped):
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13)
    folds = splitter.run_folds(df_grouped, COLS)
    assert len(folds) == N_SPLITS


def test_each_fold_valid(df_grouped):
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13)
    folds = splitter.run_folds(df_grouped, COLS)
    for res in folds:
        assert len(res.train) > 0
        assert len(res.test) > 0
        assert "fold_index" in res.stats


def test_all_ids_appear_in_test_once(df_grouped):
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13)
    folds = splitter.run_folds(df_grouped, COLS)
    all_test_ids = []
    for res in folds:
        all_test_ids.extend(res.test["id"].tolist())
    assert len(all_test_ids) == len(df_grouped)
    assert set(all_test_ids) == set(df_grouped["id"])


def test_no_overlap_per_fold(df_grouped):
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13)
    folds = splitter.run_folds(df_grouped, COLS)
    for res in folds:
        assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_leakage_zero_per_fold(df_grouped):
    """Core invariant: no group appears in both train and test within any fold."""
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="group", seed=13)
    folds = splitter.run_folds(df_grouped, COLS)
    for res in folds:
        assert res.stats["leak_groups_train_test"] == 0
        train_groups = set(res.train["group"])
        test_groups = set(res.test["group"])
        assert train_groups & test_groups == set()


def test_missing_group_col_raises(df_base):
    splitter = GroupKFoldSplitter(n_splits=N_SPLITS, group_col="NONEXISTENT")
    with pytest.raises((ValueError, KeyError)):
        splitter.run_folds(df_base, COLS)
