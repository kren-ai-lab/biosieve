from __future__ import annotations
import numpy as np

def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)
