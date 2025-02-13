from abc import ABC, abstractmethod
import numpy as np
class Pruner(ABC):
    def __init__(self):
        self.intermediate_values = np.empty((0,0))
        self.current_eval_values = np.empty(0)
        self.current_step: int = -1
        self.current_eval = 0

    @property
    def current_value(self):
        if self.current_eval_values.size == 0: return np.nan
        return self.current_eval_values[self.current_step]

    def _internal_post_eval(self):
        cur_len = self.intermediate_values.shape[1]
        if len(self.current_eval_values) > cur_len:
            self.intermediate_values = np.hstack(
                (self.intermediate_values, np.full(
                    (self.intermediate_values.shape[0], len(self.current_eval_values) - cur_len),
                    np.nan)
                 )
            )
        self.intermediate_values = np.vstack((self.intermediate_values, self.current_eval_values))
        self.current_eval_values = np.full(self.intermediate_values.shape[1], np.nan)
        self.current_step = -1
        self.post_eval()

    def post_eval(self): pass

    def _internal_report(self, step, value):
        if step <= self.current_step: self._internal_post_eval()
        else: self.current_step = step

        if step >= self.current_eval_values.size:
            # this is way faster than using np.append
            cev: list = self.current_eval_values.tolist()
            cev.extend([np.nan for i in range((step - len(cev)) + 1)])
            self.current_eval_values = np.array(cev)

        self.current_eval_values[step] = value
        self.post_report(step, value)

    def post_report(self, step, value):
        pass


    @abstractmethod
    def should_prune(self) -> bool: ...

class NoopPruner(Pruner):
    def _internal_post_eval(self): pass
    def _internal_report(self, step, value): pass
    def should_prune(self) -> bool: return False
