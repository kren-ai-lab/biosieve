"""Descriptor matrix extraction helpers for reduction strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import polars as pl


@dataclass(frozen=True)
class DescriptorMatrix:
    """In-memory descriptor matrix aligned to selected descriptor columns."""

    cols: list[str]
    X: np.ndarray  # shape (N, K)


def infer_descriptor_columns(
    df: pl.DataFrame,
    prefix: str = "desc_",
    explicit_cols: list[str] | None = None,
) -> list[str]:
    """Infer descriptor columns from an explicit list or a shared prefix."""
    if explicit_cols is not None:
        missing = [c for c in explicit_cols if c not in df.columns]
        if missing:
            msg = f"Explicit descriptor columns missing from dataframe: {missing}"
            raise ValueError(msg)
        if len(explicit_cols) == 0:
            msg = "explicit_cols was provided but empty."
            raise ValueError(msg)
        return list(explicit_cols)

    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if len(cols) == 0:
        msg = (
            f"No descriptor columns found with prefix '{prefix}'. "
            f"Provide columns with that prefix or pass explicit_cols."
        )
        raise ValueError(msg)
    return cols


def extract_descriptor_matrix(
    df: pl.DataFrame,
    cols: list[str],
    dtype: str = "float32",
) -> DescriptorMatrix:
    """Extract and validate a numeric descriptor matrix from a DataFrame."""
    X = df[cols].to_numpy()
    if X.ndim != 2:  # noqa: PLR2004
        msg = f"Descriptor matrix must be 2D. Got shape {X.shape}"
        raise ValueError(msg)

    # convert + check finite
    X = X.astype(dtype, copy=False)
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        msg = f"Descriptor matrix contains non-finite values at positions (first 5): {bad[:5].tolist()}"
        raise ValueError(msg)

    return DescriptorMatrix(cols=cols, X=X)
