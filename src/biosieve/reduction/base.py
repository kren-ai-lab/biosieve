from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
import pandas as pd
from biosieve.types import Columns

@dataclass(frozen=True)
class ReductionResult:
    df: pd.DataFrame
    mapping: pd.DataFrame
    strategy: str
    params: dict

class Reducer(Protocol):
    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult: ...
