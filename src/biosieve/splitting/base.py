from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
from biosieve.types import Columns

@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    assignments: pd.DataFrame
    strategy: str
    params: dict
    seed: int

class Splitter(Protocol):
    def run(self, df: pd.DataFrame, cols: Columns, seed: int) -> SplitResult: ...
