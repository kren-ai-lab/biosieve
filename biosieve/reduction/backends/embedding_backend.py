from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EmbeddingStore:
    ids: list[str]
    X: np.ndarray  # shape (N, D), float32/float64


def _read_ids_csv(ids_path: Path, id_col: str = "id") -> list[str]:
    df = pd.read_csv(ids_path)
    if id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        # fallback: first column
        ids = df.iloc[:, 0].astype(str).tolist()
    if len(ids) == 0:
        raise ValueError(f"No ids found in ids file: {ids_path}")
    return ids


def load_embeddings(
    embeddings_path: str,
    ids_path: str,
    ids_col: str = "id",
    dtype: str | None = None,
) -> EmbeddingStore:
    ep = Path(embeddings_path)
    ip = Path(ids_path)

    if not ep.exists():
        raise FileNotFoundError(f"Embeddings file not found: {ep}")
    if not ip.exists():
        raise FileNotFoundError(f"Embedding ids file not found: {ip}")

    X = np.load(ep)
    if X.ndim != 2:
        raise ValueError(f"Embeddings must be 2D array (N,D). Got shape {X.shape}")

    ids = _read_ids_csv(ip, id_col=ids_col)
    if len(ids) != X.shape[0]:
        raise ValueError(
            f"Mismatch: ids ({len(ids)}) vs embeddings rows ({X.shape[0]}). "
            "They must align 1-to-1 in the same order."
        )

    if dtype is not None:
        X = X.astype(dtype, copy=False)

    return EmbeddingStore(ids=ids, X=X)
