"""
Shared fixtures for BioSieve test suite.

Synthetic data is generated with a fixed RNG seed (42) so tests are
deterministic and independent of the example files in examples/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biosieve.core.strategies import build_registry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N = 50
SEED = 42
AA = "ACDEFGHIKLMNPQRSTVWY"  # standard amino acids
RNG = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _random_sequence(rng: np.random.Generator, min_len: int = 20, max_len: int = 40) -> str:
    length = int(rng.integers(min_len, max_len + 1))
    return "".join(rng.choice(list(AA), size=length))


def _make_ids(n: int) -> list[str]:
    return [f"pep_{i:03d}" for i in range(n)]


# ---------------------------------------------------------------------------
# Base DataFrame fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def df_base() -> pd.DataFrame:
    """50 rows: id + sequence. No duplicates."""
    rng = np.random.default_rng(SEED)
    ids = _make_ids(N)
    seqs = [_random_sequence(rng) for _ in range(N)]
    return pd.DataFrame({"id": ids, "sequence": seqs})


@pytest.fixture
def df_labeled(df_base: pd.DataFrame) -> pd.DataFrame:
    """df_base + binary label column (~balanced)."""
    rng = np.random.default_rng(SEED + 1)
    labels = rng.integers(0, 2, size=N).tolist()
    return df_base.assign(label=labels)


@pytest.fixture
def df_grouped(df_base: pd.DataFrame) -> pd.DataFrame:
    """df_base + group column (5 groups, 10 rows each)."""
    groups = [f"study_{c}" for c in "ABCDE"]
    group_col = [groups[i // 10] for i in range(N)]
    return df_base.assign(group=group_col)


@pytest.fixture
def df_timed(df_base: pd.DataFrame) -> pd.DataFrame:
    """df_base + date column (string, ascending from 2019 to 2024)."""
    import datetime

    start = datetime.date(2019, 1, 1)
    dates = [str(start + datetime.timedelta(days=i * 40)) for i in range(N)]
    return df_base.assign(date=dates)


@pytest.fixture
def df_clustered(df_base: pd.DataFrame) -> pd.DataFrame:
    """df_base + cluster_id column (10 clusters, 5 rows each)."""
    cluster_col = [f"clust_{(i // 5):02d}" for i in range(N)]
    return df_base.assign(cluster_id=cluster_col)


@pytest.fixture
def df_descriptors(df_base: pd.DataFrame) -> pd.DataFrame:
    """df_base + 10 numeric descriptor columns (desc_000..desc_009)."""
    rng = np.random.default_rng(SEED + 2)
    desc = rng.standard_normal((N, 10)).astype(np.float32)
    desc_df = pd.DataFrame(desc, columns=[f"desc_{i:03d}" for i in range(10)])
    return pd.concat([df_base, desc_df], axis=1)


@pytest.fixture
def df_full(df_base: pd.DataFrame) -> pd.DataFrame:
    """All columns combined: id, sequence, label, group, cluster_id, date, target, desc_*."""
    import datetime

    rng = np.random.default_rng(SEED + 3)
    groups = [f"study_{c}" for c in "ABCDE"]
    start = datetime.date(2019, 1, 1)

    extra = {
        "label": rng.integers(0, 2, size=N).tolist(),
        "group": [groups[i // 10] for i in range(N)],
        "cluster_id": [f"clust_{(i // 5):02d}" for i in range(N)],
        "date": [str(start + datetime.timedelta(days=i * 40)) for i in range(N)],
        "target": rng.random(size=N).tolist(),
    }
    desc = rng.standard_normal((N, 10)).astype(np.float32)
    desc_df = pd.DataFrame(desc, columns=[f"desc_{i:03d}" for i in range(10)])
    return pd.concat([df_base.assign(**extra), desc_df], axis=1)


# ---------------------------------------------------------------------------
# Embedding fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def embeddings_fixture(df_base: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """(N, 32) float32 embedding matrix aligned to df_base ids."""
    rng = np.random.default_rng(SEED + 4)
    X = rng.standard_normal((N, 32)).astype(np.float32)
    # L2-normalize so cosine similarity is well-defined
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1.0, norms)
    ids = df_base["id"].tolist()
    return X, ids


# ---------------------------------------------------------------------------
# Structural edges fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def edges_fixture(df_base: pd.DataFrame) -> pd.DataFrame:
    """~150 sparse (id1, id2, distance) pairs between the 50 ids."""
    rng = np.random.default_rng(SEED + 5)
    ids = df_base["id"].tolist()
    rows = []
    for i in range(N):
        # Each node gets 3 random neighbours (no self-loops)
        neighbours = rng.choice([j for j in range(N) if j != i], size=3, replace=False)
        for j in neighbours:
            if i < j:  # avoid duplicates
                rows.append({"id1": ids[i], "id2": ids[j], "distance": float(rng.random())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# File-backed fixtures (write to tmp_path)
# ---------------------------------------------------------------------------


@pytest.fixture
def csv_path(df_base: pd.DataFrame, tmp_path):
    """Write df_base to a CSV file; return the Path."""
    p = tmp_path / "dataset.csv"
    df_base.to_csv(p, index=False)
    return p


@pytest.fixture
def embeddings_files(embeddings_fixture, tmp_path):
    """Write embeddings.npy + ids.csv; return (emb_path, ids_path)."""
    X, ids = embeddings_fixture
    emb_path = tmp_path / "embeddings.npy"
    ids_path = tmp_path / "embedding_ids.csv"
    np.save(emb_path, X)
    pd.DataFrame({"id": ids}).to_csv(ids_path, index=False)
    return emb_path, ids_path


@pytest.fixture
def edges_file(edges_fixture: pd.DataFrame, tmp_path):
    """Write edges CSV; return the Path."""
    p = tmp_path / "struct_edges.csv"
    edges_fixture.to_csv(p, index=False)
    return p


@pytest.fixture
def cluster_map_file(df_clustered: pd.DataFrame, tmp_path):
    """Write (id, cluster_id) mapping CSV; return the Path."""
    p = tmp_path / "cluster_map.csv"
    df_clustered[["id", "cluster_id"]].to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Registry fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def registry():
    return build_registry()
