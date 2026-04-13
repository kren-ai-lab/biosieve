"""Tests for HomologyAwareSplitter.

Only mode="precomputed" is tested by default.
mmseqs2 mode tests are marked @pytest.mark.mmseqs2 and skipped if binary absent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    import pytest



import shutil

import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.homology_aware import HomologyAwareSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path_precomputed(df_base: pd.DataFrame, cluster_map_file: Path) -> None:
    splitter = HomologyAwareSplitter(
        mode="precomputed",
        clusters_path=str(cluster_map_file),
        clusters_format="csv",
        member_col="id",
        cluster_col="cluster_id",
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "homology_aware"
    assert len(res.train) + len(res.test) == len(df_base)
    assert len(res.train) > 0
    assert len(res.test) > 0


def test_no_overlap_precomputed(df_base: pd.DataFrame, cluster_map_file: Path) -> None:
    splitter = HomologyAwareSplitter(
        mode="precomputed",
        clusters_path=str(cluster_map_file),
        clusters_format="csv",
        member_col="id",
        cluster_col="cluster_id",
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_leakage_zero_precomputed(df_base: pd.DataFrame, cluster_map_file: Path) -> None:
    """Core invariant: no homology cluster in both train and test."""
    splitter = HomologyAwareSplitter(
        mode="precomputed",
        clusters_path=str(cluster_map_file),
        clusters_format="csv",
        member_col="id",
        cluster_col="cluster_id",
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)
    assert res.stats["leak_clusters_train_test"] == 0


def test_missing_clusters_path_raises(df_base: pd.DataFrame, tmp_path: Path) -> None:
    splitter = HomologyAwareSplitter(
        mode="precomputed",
        clusters_path=str(tmp_path / "nonexistent.csv"),
        clusters_format="csv",
        member_col="id",
        cluster_col="cluster_id",
    )
    with pytest.raises(Exception):
        splitter.run(df_base, COLS)


# ---------------------------------------------------------------------------
# mmseqs2 mode tests — require binary in PATH
# ---------------------------------------------------------------------------


@pytest.mark.mmseqs2
def test_happy_path_mmseqs2(df_base: pd.DataFrame, tmp_path: Path) -> None:
    if shutil.which("mmseqs") is None:
        pytest.skip("mmseqs binary not found in PATH")

    splitter = HomologyAwareSplitter(
        mode="mmseqs2",
        min_seq_id=0.9,
        coverage=0.8,
        threads=1,
        work_dir=str(tmp_path / "mmseqs_work"),
        keep_work=False,
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)
    assert isinstance(res, SplitResult)
    assert res.stats["leak_clusters_train_test"] == 0
