"""Tests for StratifiedKFoldSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pytest


import pytest

from biosieve.splitting.stratified_kfold import StratifiedKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


def test_stats_have_label_counts(df_labeled: pd.DataFrame) -> None:
    splitter = StratifiedKFoldSplitter(n_splits=N_SPLITS, label_col="label", seed=13)
    folds = splitter.run_folds(df_labeled, COLS)
    for res in folds:
        assert "train_label_counts" in res.stats or "test_label_counts" in res.stats


def test_missing_label_col_raises(df_base: pd.DataFrame) -> None:
    splitter = StratifiedKFoldSplitter(n_splits=N_SPLITS, label_col="NONEXISTENT")
    with pytest.raises((ValueError, KeyError)):
        splitter.run_folds(df_base, COLS)
