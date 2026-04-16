"""Structural edge loading helpers for graph-based reducers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class StructuralEdges:
    """Undirected structural neighborhood graph."""

    # adjacency: id -> list of (neighbor_id, value)
    adj: dict[str, list[tuple[str, float]]]
    n_edges: int


def load_edges_csv(
    path: str,
    id1_col: str = "id1",
    id2_col: str = "id2",
    value_col: str = "distance",
) -> StructuralEdges:
    """Load an undirected edge list into adjacency form."""
    p = Path(path)
    if not p.exists():
        msg = f"Structural edges file not found: {p}"
        raise FileNotFoundError(msg)

    df = pl.read_csv(p)
    for col in (id1_col, id2_col, value_col):
        if col not in df.columns:
            msg = f"Missing required column '{col}' in {p}. Found: {df.columns}"
            raise ValueError(msg)

    adj: dict[str, list[tuple[str, float]]] = {}
    n_edges = 0

    for row in df.iter_rows(named=True):
        a = str(row[id1_col])
        b = str(row[id2_col])
        v = float(row[value_col])

        # undirected graph assumed for structural similarity
        adj.setdefault(a, []).append((b, v))
        adj.setdefault(b, []).append((a, v))
        n_edges += 1

    return StructuralEdges(adj=adj, n_edges=n_edges)
