from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from biosieve.types import Columns


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    val: Optional[pd.DataFrame]
    strategy: str
    params: Dict[str, Any]
    stats: Dict[str, Any]


class Splitter:
    """
    Base protocol-like splitter.
    """
    @property
    def strategy(self) -> str:
        raise NotImplementedError

    def run(self, df: pd.DataFrame, cols: Columns) -> SplitResult:
        raise NotImplementedError
