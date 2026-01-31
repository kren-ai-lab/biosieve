from __future__ import annotations
import pandas as pd

def normalize_sequences(df: pd.DataFrame, seq_col: str) -> pd.DataFrame:
    out = df.copy()
    out[seq_col] = out[seq_col].astype(str).str.strip().str.upper()
    return out
