from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

from biosieve.splitting.base import SplitResult
from biosieve.splitting.config import SplitFractions
from biosieve.types import Columns

@dataclass(frozen=True)
class RandomSplit:
    fractions: SplitFractions

    strategy: str = "random"

    def run(self, df: pd.DataFrame, cols: Columns, seed: int) -> SplitResult:
        self.fractions.validate()
        work = df.copy().sort_values(cols.id_col, kind="mergesort").reset_index(drop=True)

        n = len(work)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)

        n_train = int(round(self.fractions.train * n))
        n_val = int(round(self.fractions.val * n))

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

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
            params={"fractions": {"train": self.fractions.train, "val": self.fractions.val, "test": self.fractions.test}},
            seed=seed,
        )
