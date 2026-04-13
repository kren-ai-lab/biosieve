"""Tests for TimeSplitter."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pytest


import pandas as pd
import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.time_based import TimeSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_timed: pd.DataFrame) -> None:
    splitter = TimeSplitter(time_col="date", test_size=0.2)
    res = splitter.run(df_timed, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "time"
    assert len(res.train) + len(res.test) == len(df_timed)
    assert len(res.train) > 0
    assert len(res.test) > 0


def test_no_overlap(df_timed: pd.DataFrame) -> None:
    splitter = TimeSplitter(time_col="date", test_size=0.2)
    res = splitter.run(df_timed, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_chronological_order(df_timed: pd.DataFrame) -> None:
    """All train dates must be earlier than all test dates."""
    splitter = TimeSplitter(time_col="date", test_size=0.2)
    res = splitter.run(df_timed, COLS)

    max_train = pd.to_datetime(res.train["date"]).max()
    min_test = pd.to_datetime(res.test["date"]).min()
    assert max_train <= min_test


def test_val_when_requested(df_timed: pd.DataFrame) -> None:
    splitter = TimeSplitter(time_col="date", test_size=0.2, val_size=0.1)
    res = splitter.run(df_timed, COLS)
    assert res.val is not None
    assert len(res.val) > 0


def test_missing_time_col_raises(df_base: pd.DataFrame) -> None:
    splitter = TimeSplitter(time_col="NONEXISTENT", test_size=0.2)
    with pytest.raises((ValueError, KeyError)):
        splitter.run(df_base, COLS)
