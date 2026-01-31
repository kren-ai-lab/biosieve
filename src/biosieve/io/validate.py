from __future__ import annotations
import pandas as pd
from biosieve.types import Columns

def validate_required_columns(df: pd.DataFrame, cols: Columns) -> None:
    missing = []
    for c in [cols.id_col, cols.seq_col]:
        if c not in df.columns:
            missing.append(c)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
