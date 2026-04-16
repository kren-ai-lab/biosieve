"""Protocols and result containers for reduction strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import polars as pl

    from biosieve.types import Columns

__all__ = ["Reducer", "ReductionResult"]


@dataclass(frozen=True, slots=True)
class ReductionResult:
    """Container for redundancy-reduction outputs.

    Args:
        df: The reduced (non-redundant) dataset.
        mapping:
            Optional mapping table describing which original ids were removed and which
            representative id they were assigned to.
            Recommended schema (stable columns):
            - removed_id
            - representative_id
            - cluster_id (optional)
            - score (optional; similarity/distance/identity depending on reducer)
        strategy: Strategy identifier (e.g., "mmseqs2", "embedding_cosine", "descriptor_euclidean").
        params:
            Effective parameters used by the reducer (after defaults).
            Must be JSON-serializable (or easily coercible for reporting).
        stats: Optional reducer-specific statistics (coverage, thresholds effective, etc.).

    Notes:
        - `mapping` may be None for reducers that do not produce explicit assignments.
        Runners should handle this and optionally write an empty mapping file for
        stable downstream workflows.
        - The `df` output must preserve the original `cols.id_col` values for the
        retained representatives.

    Examples:
        >>> res = reducer.run(df, cols)
        >>> res.df.shape

    """

    df: pl.DataFrame
    mapping: pl.DataFrame | None
    strategy: str
    params: dict[str, Any]
    stats: dict[str, Any] | None = None


@runtime_checkable
class Reducer(Protocol):
    """Protocol for redundancy-reduction strategies.

    Required interface:
    - `strategy` property: returns the reducer name used in reports and CLI.
    - `run(df, cols) -> ReductionResult`
    """

    @property
    def strategy(self) -> str:  # pragma: no cover
        """Return the reducer strategy identifier."""
        ...

    def run(self, df: pl.DataFrame, cols: Columns) -> ReductionResult:  # pragma: no cover
        """Run reduction and return reduced data plus mapping metadata."""
        ...
