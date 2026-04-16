"""Unit tests for reduction backend helpers.

These backends are pure data-loading / feature-extraction utilities with
well-defined error conditions. Testing them in isolation avoids relying on
the full reducer pipeline to surface loading bugs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biosieve.reduction.backends.descriptor_backend import (
    DescriptorMatrix,
    extract_descriptor_matrix,
    infer_descriptor_columns,
)
from biosieve.reduction.backends.embedding_backend import EmbeddingStore, load_embeddings

# ---------------------------------------------------------------------------
# embedding_backend
# ---------------------------------------------------------------------------


class TestLoadEmbeddings:
    def test_happy_path(self, embeddings_files: tuple[Path, Path]) -> None:
        emb_path, ids_path = embeddings_files
        store = load_embeddings(str(emb_path), str(ids_path))
        assert isinstance(store, EmbeddingStore)
        assert store.X.ndim == 2
        assert len(store.ids) == store.X.shape[0]

    def test_ids_align_with_dataframe(
        self, embeddings_files: tuple[Path, Path], df_base: pd.DataFrame
    ) -> None:
        emb_path, ids_path = embeddings_files
        store = load_embeddings(str(emb_path), str(ids_path))
        assert store.ids == df_base["id"].tolist()

    def test_missing_embeddings_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Embeddings file not found"):
            load_embeddings(
                str(tmp_path / "nope.npy"),
                str(tmp_path / "ids.csv"),
            )

    def test_missing_ids_file_raises(self, embeddings_files: tuple[Path, Path], tmp_path: Path) -> None:
        emb_path, _ = embeddings_files
        with pytest.raises(FileNotFoundError, match="Embedding ids file not found"):
            load_embeddings(str(emb_path), str(tmp_path / "nope.csv"))

    def test_1d_array_raises(self, tmp_path: Path) -> None:
        bad_emb = tmp_path / "bad.npy"
        np.save(bad_emb, np.array([1.0, 2.0, 3.0]))
        ids_path = tmp_path / "ids.csv"
        pd.DataFrame({"id": ["a", "b", "c"]}).to_csv(ids_path, index=False)
        with pytest.raises(ValueError, match="2D"):
            load_embeddings(str(bad_emb), str(ids_path))

    def test_row_count_mismatch_raises(self, tmp_path: Path) -> None:
        emb_path = tmp_path / "emb.npy"
        np.save(emb_path, np.zeros((5, 8), dtype=np.float32))
        ids_path = tmp_path / "ids.csv"
        pd.DataFrame({"id": ["a", "b", "c"]}).to_csv(ids_path, index=False)  # only 3 ids for 5 rows
        with pytest.raises(ValueError, match="Mismatch"):
            load_embeddings(str(emb_path), str(ids_path))

    def test_dtype_conversion(self, embeddings_files: tuple[Path, Path]) -> None:
        emb_path, ids_path = embeddings_files
        store = load_embeddings(str(emb_path), str(ids_path), dtype="float64")
        assert store.X.dtype == np.float64


# ---------------------------------------------------------------------------
# descriptor_backend
# ---------------------------------------------------------------------------


class TestInferDescriptorColumns:
    def test_prefix_inference(self, df_descriptors: pd.DataFrame) -> None:
        cols = infer_descriptor_columns(df_descriptors, prefix="desc_")
        assert len(cols) == 10
        assert all(c.startswith("desc_") for c in cols)

    def test_explicit_cols(self, df_descriptors: pd.DataFrame) -> None:
        explicit = ["desc_000", "desc_001"]
        cols = infer_descriptor_columns(df_descriptors, explicit_cols=explicit)
        assert cols == explicit

    def test_no_matching_prefix_raises(self, df_descriptors: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="No descriptor columns found"):
            infer_descriptor_columns(df_descriptors, prefix="NOMATCH_")

    def test_explicit_cols_missing_from_df_raises(self, df_descriptors: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="missing from dataframe"):
            infer_descriptor_columns(df_descriptors, explicit_cols=["desc_000", "NONEXISTENT"])

    def test_empty_explicit_cols_raises(self, df_descriptors: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="empty"):
            infer_descriptor_columns(df_descriptors, explicit_cols=[])


class TestExtractDescriptorMatrix:
    def test_happy_path(self, df_descriptors: pd.DataFrame) -> None:
        cols = [c for c in df_descriptors.columns if c.startswith("desc_")]
        mat = extract_descriptor_matrix(df_descriptors, cols)
        assert isinstance(mat, DescriptorMatrix)
        assert mat.X.shape == (len(df_descriptors), len(cols))
        assert mat.X.dtype == np.float32

    def test_non_finite_raises(self, df_descriptors: pd.DataFrame) -> None:
        cols = [c for c in df_descriptors.columns if c.startswith("desc_")]
        bad_df = df_descriptors.copy()
        bad_df.loc[0, cols[0]] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            extract_descriptor_matrix(bad_df, cols)

    def test_dtype_preserved(self, df_descriptors: pd.DataFrame) -> None:
        cols = [c for c in df_descriptors.columns if c.startswith("desc_")]
        mat = extract_descriptor_matrix(df_descriptors, cols, dtype="float64")
        assert mat.X.dtype == np.float64
