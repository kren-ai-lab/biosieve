"""Tests for DistanceAwareKFoldSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


from biosieve.splitting.distance_aware_kfold import DistanceAwareKFoldSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")
N_SPLITS = 3


def test_descriptors_mode(df_descriptors: pl.DataFrame) -> None:
    splitter = DistanceAwareKFoldSplitter(
        n_splits=N_SPLITS,
        feature_mode="descriptors",
        descriptor_prefix="desc_",
        seed=13,
    )
    folds = splitter.run_folds(df_descriptors, COLS)
    assert len(folds) == N_SPLITS
    for res in folds:
        assert set(res.train["id"].to_list()) & set(res.test["id"].to_list()) == set()
