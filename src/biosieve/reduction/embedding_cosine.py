from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns


@dataclass(frozen=True)
class EmbeddingCosineReducer:
    """
    Plugin backend: expects embeddings already computed (not handled by BioSieve core).
    """
    threshold: float = 0.95

    @property
    def strategy(self) -> str:
        return "embedding_cosine"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        raise NotImplementedError(
            "EmbeddingCosineReducer requires embeddings (matrix or path). "
            "Implement as plugin to compute similarity and return ReductionResult."
        )
