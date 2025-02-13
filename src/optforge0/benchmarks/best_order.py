import difflib
from typing import TYPE_CHECKING

import numpy as np

from .._types import Numeric, NumericArrayLike
from ._hundred_words import DOTCH_LIST_PLUS_3
from .benchmark import Benchmark

if TYPE_CHECKING:
    from ..trial import Trial

class BasicNumberSorting(Benchmark):
    def __init__(self, n: int = 100, log_params = True, note = None):
        self.n = n
        super().__init__(log_params = log_params, note = note)
    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        numbers = np.array(trial.suggest_permutation('numbers', self.n, init = 'uniform'),)
        return np.abs(np.diff(numbers)).sum() - (self.n-1)

class WordSimilaritySorting(Benchmark):
    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        words = trial.suggest_permutation('words', DOTCH_LIST_PLUS_3)
        total_similarity = 0
        for w, w2 in zip(words, words[1:]):
            total_similarity += 1 - difflib.SequenceMatcher(None, w, w2).ratio()
        return total_similarity