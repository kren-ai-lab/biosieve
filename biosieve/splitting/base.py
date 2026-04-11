from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import pandas as pd

from biosieve.types import Columns

__all__ = ["SplitResult", "Splitter", "KFoldSplitter"]


@dataclass(frozen=True, slots=True)
class SplitResult:
    """
    Container for split outputs.

    Parameters
    ----------
    train:
        Training split.
    test:
        Test split.
    val:
        Optional validation split. If a strategy does not create validation, this is None.
    strategy:
        Strategy identifier (e.g., "random", "group_kfold", "stratified_numeric").
    params:
        Effective parameters used by the strategy (after defaults).
        This is intended for reporting and reproducibility.
    stats:
        Strategy-specific statistics (counts, leakage checks, distribution summaries, etc.).

    Notes
    -----
    - `params` and `stats` should be JSON-serializable (or easily coercible),
      because runners typically export them as JSON reports.
    - For k-fold strategies, each fold should have `stats["fold_index"]` (int).

    Examples
    --------
    A single split strategy returns one SplitResult:

    >>> res = splitter.run(df, cols)
    >>> res.train.shape, res.test.shape

    A k-fold strategy returns multiple SplitResult objects (one per fold),
    typically via `run_folds` on a `KFoldSplitter`.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame]
    strategy: str
    params: Dict[str, Any]
    stats: Dict[str, Any]


@runtime_checkable
class Splitter(Protocol):
    """
    Protocol for single-split strategies.

    A single-split strategy produces one (train, test, optional val) partition.

    Required interface
    ------------------
    - `strategy` property: returns the strategy name used in reports and CLI.
    - `run(df, cols) -> SplitResult`
    """

    @property
    def strategy(self) -> str:  # pragma: no cover
        ...

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:  # pragma: no cover
        ...


@runtime_checkable
class KFoldSplitter(Protocol):
    """
    Protocol for k-fold strategies.

    A k-fold strategy produces a list of SplitResult objects, one per fold.
    Runners can detect this protocol (or simply check for `run_folds`) and export
    folds into `fold_00/`, `fold_01/`, etc.

    Required interface
    ------------------
    - `strategy` property
    - `run_folds(df, cols) -> list[SplitResult]`

    Notes
    -----
    - Each SplitResult returned should include `stats["fold_index"]`.
    """

    @property
    def strategy(self) -> str:  # pragma: no cover
        ...

    def run_folds(self, df: pd.DataFrame, cols: Columns) -> list[SplitResult]:  # pragma: no cover
        ...
