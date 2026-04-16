"""Tests for StratifiedNumericKFoldSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pytest


import pytest

from biosieve.splitting.stratified_numeric_kfold import StratifiedNumericKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


def test_missing_label_col_raises(df_base: pl.DataFrame) -> None:
    splitter = StratifiedNumericKFoldSplitter(n_splits=N_SPLITS, label_col="NONEXISTENT", n_bins=5)
    with pytest.raises((ValueError, KeyError)):
        splitter.run_folds(df_base, COLS)
