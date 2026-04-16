"""Shared structural invariants tested across every reducer.

Each reducer must guarantee:
  1. The mapping DataFrame, when non-empty, has the expected column schema.
  2. The partition is complete: kept | removed == all input ids, kept & removed == empty.

These properties are framework-level contracts, not strategy-specific behaviour.
Testing them once per strategy via parametrize is cleaner than copy-pasting the
same assertions into every individual test module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from biosieve.reduction.base import Reducer, ReductionResult

import pandas as pd
import pytest

from biosieve.types import Columns

COLS = Columns(id_col="id", seq_col="sequence")


# ---------------------------------------------------------------------------
# Parametrized fixture -- yields (reducer, dataframe) for each strategy
# ---------------------------------------------------------------------------


try:
    import datasketch as _datasketch_mod  # noqa: F401

    _has_datasketch = True
except ImportError:
    _has_datasketch = False


@pytest.fixture(
    params=[
        "exact",
        "identity_greedy",
        "kmer_jaccard",
        pytest.param(
            "minhash_jaccard",
            marks=pytest.mark.skipif(
                not _has_datasketch,
                reason="datasketch not installed",
            ),
        ),
        "descriptor_euclidean",
        "structural_distance",
        "embedding_cosine",
    ]
)
def reducer_and_df(
    request: pytest.FixtureRequest,
    df_base: pd.DataFrame,
    df_descriptors: pd.DataFrame,
    embeddings_files: tuple[Path, Path],
    edges_file: Path,
) -> tuple[Reducer, pd.DataFrame]:
    """Return (reducer_instance, input_df) for every non-mmseqs2 reducer."""
    from biosieve.reduction.descriptor_euclidean import DescriptorEuclideanReducer
    from biosieve.reduction.embedding_cosine import EmbeddingCosineReducer
    from biosieve.reduction.exact import ExactDedupReducer
    from biosieve.reduction.identity_greedy import IdentityGreedyReducer
    from biosieve.reduction.kmer_jaccard import KmerJaccardReducer
    from biosieve.reduction.minhash_jaccard import MinHashJaccardReducer
    from biosieve.reduction.structural_distance import StructuralDistanceReducer

    emb_path, ids_path = embeddings_files
    dupes = df_base.head(3).copy()
    dupes["id"] = ["dup_000", "dup_001", "dup_002"]
    df_with_dupes = pd.concat([df_base, dupes], ignore_index=True)

    cases: dict[str, Any] = {
        "exact": (ExactDedupReducer(), df_with_dupes),
        "identity_greedy": (IdentityGreedyReducer(threshold=0.5), df_base),
        "kmer_jaccard": (KmerJaccardReducer(threshold=0.3, k=3), df_base),
        "minhash_jaccard": (MinHashJaccardReducer(threshold=0.3, k=3, num_perm=64), df_base),
        "descriptor_euclidean": (
            DescriptorEuclideanReducer(threshold=2.0, descriptor_prefix="desc_"),
            df_descriptors,
        ),
        "structural_distance": (
            StructuralDistanceReducer(edges_path=str(edges_file), threshold=0.4),
            df_base,
        ),
        "embedding_cosine": (
            EmbeddingCosineReducer(
                embeddings_path=str(emb_path),
                ids_path=str(ids_path),
                threshold=0.5,
                use_faiss=False,
            ),
            df_base,
        ),
    }
    return cases[request.param]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------


def test_mapping_schema(reducer_and_df: tuple[Reducer, pd.DataFrame]) -> None:
    """Non-empty mapping must have 'removed_id' and 'representative_id' columns."""
    reducer, df = reducer_and_df
    res: ReductionResult = reducer.run(df, COLS)
    if res.mapping is not None and len(res.mapping) > 0:
        assert "removed_id" in res.mapping.columns
        assert "representative_id" in res.mapping.columns


def test_no_ids_lost(reducer_and_df: tuple[Reducer, pd.DataFrame]) -> None:
    """Partition completeness: kept | removed == all_ids, kept & removed == empty."""
    reducer, df = reducer_and_df
    res: ReductionResult = reducer.run(df, COLS)
    if res.mapping is None or len(res.mapping) == 0:
        # Nothing removed -- the kept set must equal the full input.
        assert set(res.df["id"].astype(str)) == set(df["id"].astype(str))
        return

    kept = set(res.df["id"].astype(str))
    removed = set(res.mapping["removed_id"].astype(str))
    all_ids = set(df["id"].astype(str))
    assert kept & removed == set(), "A sample appears in both kept and removed"
    assert kept | removed == all_ids, "Some samples are neither kept nor in the mapping"
