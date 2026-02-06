from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type

from biosieve.reduction.base import Reducer
from biosieve.splitting.base import Splitter


__all__ = ["StrategyRegistry"]


@dataclass(frozen=True, slots=True)
class StrategyRegistry:
    """
    Registry of available BioSieve strategies.

    The registry stores *strategy classes* (not instances). Runners/CLI can then
    instantiate them via `instantiate_strategy(...)` with validated parameters.

    Parameters
    ----------
    reducers:
        Mapping from reducer name to reducer class.
        Example: {"embedding_cosine": EmbeddingCosineReducer}
    splitters:
        Mapping from splitter name to splitter class.
        Example: {"group_kfold": GroupKFoldSplitter}

    Notes
    -----
    - The values are strategy *types* (classes), not constructed objects.
    - Strategy names should be stable because they appear in CLI and reports.
    - Keeping the registry lightweight allows `biosieve info` to list strategies
      without importing heavy optional dependencies (prefer lazy imports inside
      the strategy implementations).

    Examples
    --------
    >>> registry = StrategyRegistry(
    ...     reducers={"embedding_cosine": EmbeddingCosineReducer},
    ...     splitters={"random": RandomSplitter},
    ... )
    >>> reducer_cls = registry.get_reducer("embedding_cosine")
    """
    reducers: Dict[str, Type[Reducer]]
    splitters: Dict[str, Type[Splitter]]

    def get_reducer(self, name: str) -> Type[Reducer]:
        """
        Get a reducer class by name.

        Parameters
        ----------
        name:
            Registered reducer strategy name.

        Returns
        -------
        Type[Reducer]
            Reducer class associated with `name`.

        Raises
        ------
        ValueError
            If `name` is not registered.
        """
        if name not in self.reducers:
            raise ValueError(
                f"Unknown reduction strategy '{name}'. Available: {sorted(self.reducers)}"
            )
        return self.reducers[name]

    def get_splitter(self, name: str) -> Type[Splitter]:
        """
        Get a splitter class by name.

        Parameters
        ----------
        name:
            Registered split strategy name.

        Returns
        -------
        Type[Splitter]
            Splitter class associated with `name`.

        Raises
        ------
        ValueError
            If `name` is not registered.
        """
        if name not in self.splitters:
            raise ValueError(
                f"Unknown split strategy '{name}'. Available: {sorted(self.splitters)}"
            )
        return self.splitters[name]
