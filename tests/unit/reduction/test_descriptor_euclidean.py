"""Tests for DescriptorEuclideanReducer."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pytest


import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.descriptor_euclidean import DescriptorEuclideanReducer
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_descriptors: pl.DataFrame) -> None:
    reducer = DescriptorEuclideanReducer(
        threshold=2.0,
        descriptor_prefix="desc_",
        n_jobs=1,
    )
    res = reducer.run(df_descriptors, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "descriptor_euclidean"
    assert res.df.height <= df_descriptors.height
    assert res.df.height > 0
    assert set(res.df["id"].to_list()).issubset(set(df_descriptors["id"].to_list()))


def test_zero_threshold_removes_nothing(df_descriptors: pl.DataFrame) -> None:
    """threshold=0.0 → no pair within 0 euclidean distance → nothing removed."""
    reducer = DescriptorEuclideanReducer(threshold=0.0, descriptor_prefix="desc_")
    res = reducer.run(df_descriptors, COLS)
    assert res.df.height == df_descriptors.height


def test_no_matching_prefix_raises(df_descriptors: pl.DataFrame) -> None:
    """When descriptor_prefix matches no columns, should raise."""
    reducer = DescriptorEuclideanReducer(descriptor_prefix="NOMATCH_")
    with pytest.raises((ValueError, Exception)):
        reducer.run(df_descriptors, COLS)
