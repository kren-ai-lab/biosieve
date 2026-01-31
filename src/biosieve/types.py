from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Columns:
    id_col: str = "id"
    seq_col: str = "sequence"
    label_col: Optional[str] = "label"
    group_col: Optional[str] = None
    cluster_col: Optional[str] = None
    date_col: Optional[str] = None
