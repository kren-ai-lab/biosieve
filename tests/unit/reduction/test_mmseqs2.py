"""Tests for MMseqs2Reducer — skipped unless mmseqs binary is in PATH."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import polars as pl
    import pytest


import shutil

import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.mmseqs2 import MMseqs2Reducer
from biosieve.types import Columns

pytestmark = pytest.mark.mmseqs2

COLS = Columns(id_col="id", seq_col="sequence")


@pytest.fixture(autouse=True)
def require_mmseqs() -> None:
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs binary not found in PATH")


def test_happy_path(df_base: pl.DataFrame, tmp_path: Path) -> None:
    reducer = MMseqs2Reducer(
        min_seq_id=0.9,
        coverage=0.8,
        threads=1,
        tmp_root=str(tmp_path / "mmseqs_work"),
        keep_tmp=False,
    )
    res = reducer.run(df_base, COLS)

    assert isinstance(res, ReductionResult)
    assert res.strategy == "mmseqs2"
    assert res.df.height <= df_base.height
    assert res.df.height > 0
    assert set(res.df["id"].to_list()).issubset(set(df_base["id"].to_list()))


def test_missing_sequence_col(df_base: pl.DataFrame) -> None:
    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = MMseqs2Reducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)
