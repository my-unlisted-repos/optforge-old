import numpy as np

from .base_pruner import Pruner


class MedianPruner(Pruner):
    def __init__(self, start_step = 2, start_eval = 3, threshold=0.5):
        super().__init__()
        self.start_step = start_step
        self.threshold = threshold
        self.start_eval = start_eval
        self.percentiles = np.empty((0))

    def post_eval(self):
        self.percentiles = np.nanpercentile(self.intermediate_values, self.threshold * 100, axis=0)

    def should_prune(self) -> bool:
        if self.current_step < self.start_step: return False
        if self.current_eval < self.start_eval: return False
        if self.current_step >= self.percentiles.size: return False

        if self.percentiles[self.current_step] < self.current_value: return True
        return False
