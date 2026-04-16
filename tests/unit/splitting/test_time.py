"""Tests for TimeSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl
    import pytest

import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.time_based import TimeSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_timed: pl.DataFrame) -> None:
    splitter = TimeSplitter(time_col="date", test_size=0.2)
    res = splitter.run(df_timed, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "time"
    assert res.train.height + res.test.height == df_timed.height
    assert res.train.height > 0
    assert res.test.height > 0


def test_chronological_order(df_timed: pl.DataFrame) -> None:
    """All train dates must be earlier than all test dates."""
    splitter = TimeSplitter(time_col="date", test_size=0.2)
    res = splitter.run(df_timed, COLS)

    max_train = res.train["date"].str.to_date().max()
    min_test = res.test["date"].str.to_date().min()
    assert max_train <= min_test


def test_val_when_requested(df_timed: pl.DataFrame) -> None:
    splitter = TimeSplitter(time_col="date", test_size=0.2, val_size=0.1)
    res = splitter.run(df_timed, COLS)
    assert res.val is not None
    assert res.val.height > 0


def test_missing_time_col_raises(df_base: pl.DataFrame) -> None:
    splitter = TimeSplitter(time_col="NONEXISTENT", test_size=0.2)
    with pytest.raises((ValueError, KeyError)):
        splitter.run(df_base, COLS)
