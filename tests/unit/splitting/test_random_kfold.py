"""Tests for RandomKFoldSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:

    import pandas as pd



from biosieve.splitting.random_kfold import RandomKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


def test_returns_n_folds(df_base: pd.DataFrame) -> None:
    splitter = RandomKFoldSplitter(n_splits=N_SPLITS, seed=13)
    folds = splitter.run_folds(df_base, COLS)
    assert len(folds) == N_SPLITS


def test_each_fold_valid(df_base: pd.DataFrame) -> None:
    splitter = RandomKFoldSplitter(n_splits=N_SPLITS, seed=13)
    folds = splitter.run_folds(df_base, COLS)
    for res in folds:
        assert len(res.train) > 0
        assert len(res.test) > 0
        assert "fold_index" in res.stats


def test_all_ids_appear_in_test_once(df_base: pd.DataFrame) -> None:
    """Each sample should appear in exactly one test fold."""
    splitter = RandomKFoldSplitter(n_splits=N_SPLITS, seed=13)
    folds = splitter.run_folds(df_base, COLS)
    all_test_ids = []
    for res in folds:
        all_test_ids.extend(res.test["id"].tolist())
    assert len(all_test_ids) == len(df_base)
    assert set(all_test_ids) == set(df_base["id"])


def test_no_overlap_per_fold(df_base: pd.DataFrame) -> None:
    splitter = RandomKFoldSplitter(n_splits=N_SPLITS, seed=13)
    folds = splitter.run_folds(df_base, COLS)
    for res in folds:
        assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_val_when_requested(df_base: pd.DataFrame) -> None:
    splitter = RandomKFoldSplitter(n_splits=N_SPLITS, val_size=0.1, seed=13)
    folds = splitter.run_folds(df_base, COLS)
    for res in folds:
        assert res.val is not None
        assert len(res.val) > 0
