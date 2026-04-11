from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def validate(self) -> None:
        s = float(self.train + self.val + self.test)
        if not np.isclose(s, 1.0):
            raise ValueError(f"train+val+test must sum to 1.0, got {s}")
        for x in (self.train, self.val, self.test):
            if x < 0:
                raise ValueError("Split fractions must be non-negative")
