from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DescriptorMatrix:
    cols: List[str]
    X: np.ndarray  # shape (N, K)


def infer_descriptor_columns(
    df: pd.DataFrame,
    prefix: str = "desc_",
    explicit_cols: Optional[List[str]] = None,
) -> List[str]:
    if explicit_cols is not None:
        missing = [c for c in explicit_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Explicit descriptor columns missing from dataframe: {missing}")
        if len(explicit_cols) == 0:
            raise ValueError("explicit_cols was provided but empty.")
        return list(explicit_cols)

    cols = [c for c in df.columns if str(c).startswith(prefix)]
    if len(cols) == 0:
        raise ValueError(
            f"No descriptor columns found with prefix '{prefix}'. "
            f"Provide columns with that prefix or pass explicit_cols."
        )
    return cols


def extract_descriptor_matrix(
    df: pd.DataFrame,
    cols: List[str],
    dtype: str = "float32",
) -> DescriptorMatrix:
    X = df[cols].to_numpy()
    if X.ndim != 2:
        raise ValueError(f"Descriptor matrix must be 2D. Got shape {X.shape}")

    # convert + check finite
    X = X.astype(dtype, copy=False)
    if not np.isfinite(X).all():
        bad = np.argwhere(~np.isfinite(X))
        raise ValueError(f"Descriptor matrix contains non-finite values at positions (first 5): {bad[:5].tolist()}")

    return DescriptorMatrix(cols=cols, X=X)
