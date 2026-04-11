"""Tests for MMseqs2Reducer — skipped unless mmseqs binary is in PATH."""

from __future__ import annotations

import shutil

import pytest

from biosieve.reduction.base import ReductionResult
from biosieve.reduction.mmseqs2 import MMseqs2Reducer
from biosieve.types import Columns

pytestmark = pytest.mark.mmseqs2

COLS = Columns(id_col="id", seq_col="sequence")


@pytest.fixture(autouse=True)
def require_mmseqs():
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs binary not found in PATH")


def test_happy_path(df_base, tmp_path):
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
    assert len(res.df) <= len(df_base)
    assert len(res.df) > 0
    assert set(res.df["id"]).issubset(set(df_base["id"]))


def test_mapping_schema(df_base, tmp_path):
    reducer = MMseqs2Reducer(
        min_seq_id=0.9,
        threads=1,
        tmp_root=str(tmp_path / "mmseqs_work2"),
    )
    res = reducer.run(df_base, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        assert "removed_id" in res.mapping.columns
        assert "representative_id" in res.mapping.columns


def test_missing_sequence_col(df_base):
    bad_cols = Columns(id_col="id", seq_col="NONEXISTENT")
    reducer = MMseqs2Reducer()
    with pytest.raises((ValueError, KeyError)):
        reducer.run(df_base, bad_cols)
