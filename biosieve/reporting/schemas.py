from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(frozen=True)
class ReductionReport:
    strategy: str
    n_input: int
    n_output: int
    n_removed: int
    params: Dict[str, Any]
    seed: Optional[int] = None

@dataclass(frozen=True)
class SplitReport:
    strategy: str
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    params: Dict[str, Any]
    seed: int
