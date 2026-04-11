from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class StructuralEdges:
    # adjacency: id -> list of (neighbor_id, value)
    adj: Dict[str, List[Tuple[str, float]]]
    n_edges: int


def load_edges_csv(
    path: str,
    id1_col: str = "id1",
    id2_col: str = "id2",
    value_col: str = "distance",
) -> StructuralEdges:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Structural edges file not found: {p}")

    df = pd.read_csv(p)
    for col in (id1_col, id2_col, value_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {p}. Found: {df.columns.tolist()}")

    adj: Dict[str, List[Tuple[str, float]]] = {}
    n_edges = 0

    for _, row in df.iterrows():
        a = str(row[id1_col])
        b = str(row[id2_col])
        v = float(row[value_col])

        # undirected graph assumed for structural similarity
        adj.setdefault(a, []).append((b, v))
        adj.setdefault(b, []).append((a, v))
        n_edges += 1

    return StructuralEdges(adj=adj, n_edges=n_edges)
