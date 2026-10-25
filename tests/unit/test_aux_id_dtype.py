"""Ids in auxiliary files keep leading zeros."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from biosieve.splitting.homology_aware import HomologyAwareSplitter
from biosieve.types import Columns

if TYPE_CHECKING:
    from pathlib import Path


def test_cluster_map_ids_with_leading_zeros(tmp_path: Path) -> None:
    p = tmp_path / "clusters.csv"
    p.write_text("member_id,cluster_id\n001,07\n002,07\n003,08\n004,08\n")
    splitter = HomologyAwareSplitter(
        mode="precomputed", clusters_path=str(p), clusters_format="csv", test_size=0.5
    )
    res = splitter.run(pl.DataFrame({"id": ["001", "002", "003", "004"]}), Columns(id_col="id"))
    assert res.stats["n_clusters_total"] == 2
    assert res.stats["leak_clusters_train_test"] == 0
