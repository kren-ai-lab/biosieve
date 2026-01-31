from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from biosieve.splitting.base import SplitResult
from biosieve.splitting.config import SplitFractions
from biosieve.types import Columns

@dataclass(frozen=True)
class StratifiedSplit:
    fractions: SplitFractions
    label_col: str = "label"

    strategy: str = "stratified"

    def run(self, df: pd.DataFrame, cols: Columns, seed: int) -> SplitResult:
        self.fractions.validate()
        if self.label_col not in df.columns:
            raise ValueError(f"label_col '{self.label_col}' not found in dataframe.")

        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)
        y = work[self.label_col].astype(str).values
        idx = np.arange(len(work))

        train_idx, tmp_idx = train_test_split(
            idx,
            test_size=(1.0 - self.fractions.train),
            random_state=seed,
            stratify=y,
        )

        if self.fractions.val == 0:
            val_idx = np.array([], dtype=int)
            test_idx = tmp_idx
        elif self.fractions.test == 0:
            val_idx = tmp_idx
            test_idx = np.array([], dtype=int)
        else:
            tmp_y = y[tmp_idx]
            val_frac_of_tmp = self.fractions.val / (self.fractions.val + self.fractions.test)
            val_idx, test_idx = train_test_split(
                tmp_idx,
                test_size=(1.0 - val_frac_of_tmp),
                random_state=seed,
                stratify=tmp_y,
            )

        train = work.iloc[train_idx].reset_index(drop=True)
        val = work.iloc[val_idx].reset_index(drop=True)
        test = work.iloc[test_idx].reset_index(drop=True)

        assignments = pd.DataFrame(
            {
                cols.id_col: pd.concat([train[cols.id_col], val[cols.id_col], test[cols.id_col]]).astype(str),
                "split": (["train"] * len(train)) + (["val"] * len(val)) + (["test"] * len(test)),
                "strategy": self.strategy,
                "seed": seed,
            }
        ).reset_index(drop=True)

        return SplitResult(
            train=train, val=val, test=test,
            assignments=assignments,
            strategy=self.strategy,
            params={
                "fractions": {"train": self.fractions.train, "val": self.fractions.val, "test": self.fractions.test},
                "label_col": self.label_col,
            },
            seed=seed,
        )
