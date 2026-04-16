"""Tests for RandomKFoldSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


from biosieve.splitting.random_kfold import RandomKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


def test_val_when_requested(df_base: pd.DataFrame) -> None:
    splitter = RandomKFoldSplitter(n_splits=N_SPLITS, val_size=0.1, seed=13)
    folds = splitter.run_folds(df_base, COLS)
    for res in folds:
        assert res.val is not None
        assert len(res.val) > 0
