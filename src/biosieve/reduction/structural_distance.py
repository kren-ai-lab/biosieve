from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns


@dataclass(frozen=True)
class StructuralDistanceReducer:
    """
    Plugin backend: requires structures or a distance matrix (TM-score/RMSD/contact-map distance).
    """
    threshold: float = 0.5  # interpretation depends on metric

    @property
    def strategy(self) -> str:
        return "structural_distance"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        raise NotImplementedError(
            "StructuralDistanceReducer requires structures or precomputed structural distances. "
            "Implement as plugin and return ReductionResult."
        )
