"""Tests for ClusterAwareSplitter — includes leakage invariant."""

from __future__ import annotations

import pytest

from biosieve.splitting.base import SplitResult
from biosieve.splitting.cluster import ClusterAwareSplitter
from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


def test_happy_path(df_clustered):
    splitter = ClusterAwareSplitter(cluster_col="cluster_id", test_size=0.2, seed=13)
    res = splitter.run(df_clustered, COLS)

    assert isinstance(res, SplitResult)
    assert res.strategy == "cluster_aware"
    assert len(res.train) + len(res.test) == len(df_clustered)
    assert len(res.train) > 0
    assert len(res.test) > 0


def test_no_overlap(df_clustered):
    splitter = ClusterAwareSplitter(cluster_col="cluster_id", test_size=0.2, seed=13)
    res = splitter.run(df_clustered, COLS)
    assert set(res.train["id"]) & set(res.test["id"]) == set()


def test_leakage_zero(df_clustered):
    """Core invariant: no cluster appears in both train and test."""
    splitter = ClusterAwareSplitter(cluster_col="cluster_id", test_size=0.2, seed=13)
    res = splitter.run(df_clustered, COLS)

    assert res.stats["leak_clusters_train_test"] == 0


def test_with_mapping_file(df_base, cluster_map_file):
    """When cluster_col absent in df but cluster_map_path provided, still works."""
    splitter = ClusterAwareSplitter(
        cluster_map_path=str(cluster_map_file),
        map_id_col="id",
        map_cluster_col="cluster_id",
        test_size=0.2,
        seed=13,
    )
    res = splitter.run(df_base, COLS)
    assert len(res.train) + len(res.test) == len(df_base)
    assert res.stats["leak_clusters_train_test"] == 0


def test_missing_cluster_col_raises(df_base):
    splitter = ClusterAwareSplitter(cluster_col="NONEXISTENT", test_size=0.2)
    with pytest.raises((ValueError, KeyError)):
        splitter.run(df_base, COLS)
