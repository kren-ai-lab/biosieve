"""Tests for StratifiedNumericKFoldSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pytest


import pytest

from biosieve.splitting.stratified_numeric_kfold import StratifiedNumericKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


def test_returns_n_folds(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="target", n_bins=5, seed=13)
    folds = splitter.run_folds(df_full, COLS)
    assert len(folds) == N_SPLITS


def test_each_fold_valid(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="target", n_bins=5, seed=13)
    folds = splitter.run_folds(df_full, COLS)
    for res in folds:
        assert len(res.train) > 0
        assert len(res.test) > 0
        assert "fold_index" in res.stats


def test_all_ids_appear_in_test_once(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="target", n_bins=5, seed=13)
    folds = splitter.run_folds(df_full, COLS)
    all_test_ids = []
    for res in folds:
        all_test_ids.extend(res.test["id"].tolist())
    assert len(all_test_ids) == len(df_full)
    assert set(all_test_ids) == set(df_full["id"])


def test_no_overlap_per_fold(df_full: pd.DataFrame) -> None:
    splitter = StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="target", n_bins=5, seed=13)
    folds = splitter.run_folds(df_full, COLS)
    for res in folds:
        assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_missing_label_col_raises(df_base: pd.DataFrame) -> None:
    splitter = StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="NONEXISTENT", n_bins=5)
    with pytest.raises((ValueError, KeyError)):
        splitter.run_folds(df_base, COLS)
