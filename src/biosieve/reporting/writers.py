from __future__ import annotations
import json
from dataclasses import asdict
from typing import Any
import pandas as pd

def write_json(obj: Any, path: str) -> None:
    payload = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def write_assignments(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
