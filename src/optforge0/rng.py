from typing import Optional
import random
import numpy as np
class RNG:
    def __init__(self, seed: Optional[int]):
        self.seed = seed
        self.random = random.Random(seed)
        self.numpy = np.random.default_rng(seed)

    def copy(self):
        return RNG(self.seed)