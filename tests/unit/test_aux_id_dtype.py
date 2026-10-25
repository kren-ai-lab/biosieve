"""Ids in auxiliary files (cluster maps, edges, embedding ids) keep leading zeros."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from biosieve.cli.validate_inputs import _check_edges, _check_embeddings_alignment
from biosieve.reduction.backends.structure_backend import load_edges_csv
from biosieve.splitting.cluster import _load_cluster_map_csv
from biosieve.splitting.homology_aware import HomologyAwareSplitter, _load_mmseqs_cluster_tsv
from biosieve.types import Columns

if TYPE_CHECKING:
    from pathlib import Path

DF = pl.DataFrame({"id": ["001", "002"]})


def test_cluster_map_csv(tmp_path: Path) -> None:
    p = tmp_path / "clusters.csv"
    p.write_text("id,cluster_id\n001,07\n002,07\n")
    assert _load_cluster_map_csv(str(p), "id", "cluster_id") == {"001": "07", "002": "07"}


def test_edges_csv(tmp_path: Path) -> None:
    p = tmp_path / "edges.csv"
    p.write_text("id1,id2,distance\n001,002,0.5\n")
    assert set(load_edges_csv(str(p)).adj) == {"001", "002"}


def test_mmseqs_cluster_tsv(tmp_path: Path) -> None:
    p = tmp_path / "clusters.tsv"
    p.write_text("001\t001\n001\t002\n")
    assert _load_mmseqs_cluster_tsv(p)["member_id"].to_list() == ["001", "002"]


def test_homology_precomputed_csv(tmp_path: Path) -> None:
    p = tmp_path / "clusters.csv"
    p.write_text("member_id,cluster_id\n001,07\n002,07\n003,08\n004,08\n")
    splitter = HomologyAwareSplitter(
        mode="precomputed", clusters_path=str(p), clusters_format="csv", test_size=0.5
    )
    res = splitter.run(pl.DataFrame({"id": ["001", "002", "003", "004"]}), Columns(id_col="id"))
    assert res.stats["n_clusters_total"] == 2
    assert res.stats["leak_clusters_train_test"] == 0


def test_validate_inputs_edges(tmp_path: Path) -> None:
    p = tmp_path / "edges.csv"
    p.write_text("id1,id2,distance\n001,002,0.5\n")
    ok, _ = _check_edges(DF, "id", str(p), "id1", "id2", "distance")
    assert ok


def test_validate_inputs_embedding_ids(tmp_path: Path) -> None:
    ids = tmp_path / "ids.csv"
    ids.write_text("id\n001\n002\n")
    emb = tmp_path / "emb.npy"
    np.save(emb, np.zeros((2, 3)))
    ok, msg = _check_embeddings_alignment(DF, "id", str(emb), str(ids), "id")
    assert ok
    assert "coverage=1.000" in msg
