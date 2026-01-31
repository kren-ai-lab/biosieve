from __future__ import annotations
import pandas as pd
from biosieve.reduction.base import ReductionResult
from biosieve.types import Columns

class ExactDedupReducer:
    strategy = "exact"

    def run(self, df: pd.DataFrame, cols: Columns) -> ReductionResult:
        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
        dup = work.duplicated(subset=[cols.seq_col], keep="first")

        kept = work.loc[~dup].reset_index(drop=True)
        removed = work.loc[dup, [cols.id_col, cols.seq_col]].copy()

        rep_by_seq = kept.set_index(cols.seq_col)[cols.id_col].to_dict()
        mapping = pd.DataFrame(
            {
                "removed_id": removed[cols.id_col].astype(str).values,
                "representative_id": removed[cols.seq_col].map(rep_by_seq).astype(str).values,
            }
        )
        return ReductionResult(df=kept, mapping=mapping, strategy=self.strategy, params={})
