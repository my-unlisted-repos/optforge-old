from collections.abc import Sequence, Callable
from typing import Optional, TYPE_CHECKING, Literal

from .optimizer import Optimizer
from ..scheduler import SchedulableInt

if TYPE_CHECKING:
    from ..study import Study

__all__ = [
    "RandomRestart",
]
class OutOfBudget(Exception): pass

class OutOfBudgetCallback:
    def __init__(self, max_evals):
        self.max_evals = max_evals

    def __call__(self, study: "Study", evaluated_trial, finished_trial):
        if study.current_eval > self.max_evals:
            raise OutOfBudget()


class RandomRestart(Optimizer):
    """Random restart."""
    def __init__(
        self,
        optimizer_cls: Callable[..., Optimizer],
        evals: SchedulableInt,
        start: Literal["best", "last", "random"] = "random",
        budget=None,
    ):
        """Chains optimizers, so that each optimizer is used for a certain number of evaluations.

        :param splits: Splits, either relative to the total budget (float) or absolute (int). Defaults to None, which splits evenly.
        :param start: Which params to use when the switching to the next optimizer, defaults to "best"
        :param budget: Budget, if not set, this may be inferred from `max_evals`. Defaults to None
        """
        super().__init__()
        self.optimizer_cls = optimizer_cls
        self.evals = self.schedule(evals)
        self.budget = budget
        self.start = start

        self.cur = None
        self.idx = -1

        self._no_tell = False


    def _restart(self, study: "Study"):
        if self.cur is not None:
            if self.start == 'last': self.set_params(study.evaluated_trial.params)
            elif self.start == 'random': 
                self.params.randomize()
            elif self.start == 'best': self.params.update(study.best_trial.paramdict)
            else: raise ValueError(f'Invalid start - "{self.start}"')

        self.idx += 1
        self.cur = self.optimizer_cls()
        self.cur.set_params(self.params)

        evals: float | int = self.evals()
        if isinstance(evals, float):
            if self.budget is None: raise ValueError("Please specify budget with relative (float) evals.")
            evals = int(self.budget * evals)
        self.cur.set_budget(self.evals) # type:ignore

        for i in self.cur._schedulers:
            i.init_eval = study.current_eval if self.idx != 0 else 0 # type:ignore

        self.CONFIG = dict(self.cur.CONFIG).copy()
        if self.start == 'best': self.CONFIG['store_paramdicts'] = 'best'

        self.callbacks = [OutOfBudgetCallback(study.current_eval + evals)]

    def ask(self, study):
        if self.cur is None: self._restart(study)
        try:
            yield from self.cur.ask(study) # type:ignore
        except OutOfBudget:
            self._no_tell = True
            self._restart(study)

    def tell(self, trials, study):
        if self._no_tell:
            self._no_tell = False
            return
        try:
            if self.cur is None: self._restart(study)
            if self.cur is None: raise ValueError("Please ask before telling.")
            self.cur.tell(trials, study)
        except OutOfBudget:
            if len(trials) > 0: self._restart(study)

    def tell_not_asked(self, trial, study):
        if self._no_tell:
            self._no_tell = False
            return
        try:
            if self.cur is None: self._restart(study)
            if self.cur is None: raise ValueError("Please ask before telling.")
            self.cur.tell_not_asked(trial, study)
        except OutOfBudget:
            self._restart(study)

    def step(self, study):
        if self.cur is None: self._restart(study)
        try: self.cur.step(study) # type:ignore
        except OutOfBudget: self._restart(study)


