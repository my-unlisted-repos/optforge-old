import numpy as np

from .base_pruner import Pruner


class ThresholdPruner(Pruner):
    def __init__(self, threshold, start_step = 2):
        super().__init__()
        self.start_step = start_step
        self.threshold = threshold

    def should_prune(self) -> bool:
        if self.current_step < self.start_step: return False

        if self.current_value > self.threshold: return True
        return False
