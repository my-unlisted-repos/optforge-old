from collections.abc import Sequence
from typing import Optional, TYPE_CHECKING, Literal

from .optimizer import Optimizer
if TYPE_CHECKING:
    from ..study import Study

__all__ = [
    "Chain",
]

class OutOfBudget(Exception): pass

class OutOfBudgetCallback:
    def __init__(self, max_evals):
        self.max_evals = max_evals

    def __call__(self, study: "Study", evaluated_trial, finished_trial):
        if study.current_eval > self.max_evals:
            raise OutOfBudget()

class Chain(Optimizer):
    """Chain optimizers."""
    def __init__(
        self,
        *optimizers: Optimizer,
        splits: Optional[Sequence[int] | Sequence[float]] = None,
        start: Literal["best", "last", "random"] = "best",
        budget=None,
    ):
        """Chains optimizers, so that each optimizer is used for a certain number of evaluations.

        :param splits: Splits, either relative to the total budget (float) or absolute (int). Defaults to None, which splits evenly.
        :param start: Which params to use when the switching to the next optimizer, defaults to "best"
        :param budget: Budget, if not set, this may be inferred from `max_evals`. Defaults to None
        """
        super().__init__()
        self.optimizers = optimizers
        self.splits = splits
        self.budget = budget
        self.start = start

        self.cur = None
        self.idx = -1

        self._no_tell = False
        if self.start == 'best': self.CONFIG['store_paramdicts'] = 'best'

    def _move_to_next_optimizer(self, study: "Study"):
        if self.cur is not None:
            if self.start == 'last': self.set_params(study.evaluated_trial.params)
            elif self.start == 'random': self.params.randomize()
            elif self.start == 'best': self.params.update(study.best_trial.paramdict)
            else: raise ValueError(f'Invalid start - "{self.start}"')

        self.idx += 1
        self.cur = self.optimizers[self.idx]
        self.cur.set_params(self.params)

        if self.idx == len(self.splits):  # type:ignore
            budget = self.budget - self.splits[self.idx - 1] # type:ignore
            max_eval = self.budget
        else:
            budget = self.splits[self.idx] - (self.splits[self.idx - 1] if self.idx != 0 else 0) # type:ignore
            max_eval = self.splits[self.idx] # type:ignore
        self.cur.set_budget(budget) # type:ignore

        for i in self.cur._schedulers:
            i.init_eval = self.splits[self.idx - 1] if self.idx != 0 else 0 # type:ignore

        self.CONFIG = dict(self.cur.CONFIG).copy()
        if self.start == 'best': self.CONFIG['store_paramdicts'] = 'best'

        self.callbacks = [OutOfBudgetCallback(max_eval)]

    def _initialize_splits(self, study: "Study"):
        # budget is guaranteed to be set there if max evals is specified on study
        if self.splits is None:
            if self.budget is None: raise ValueError("Please specify budget or splits.")
            split = self.budget // len(self.optimizers)
            self.splits = [split * i for i in range(1, len(self.optimizers))]

        if isinstance(self.splits[0], float):
            if self.budget is None: raise ValueError("Please specify budget with relative (float) splits.")
            self.splits = [int(self.budget * s) for s in self.splits]


    def ask(self, study):
        self._initialize_splits(study)
        if self.cur is None: self._move_to_next_optimizer(study)
        try:
            yield from self.cur.ask(study) # type:ignore
        except OutOfBudget:
            self._no_tell = True
            self._move_to_next_optimizer(study)

    def tell(self, trials, study):
        if self._no_tell:
            self._no_tell = False
            return
        try:
            if self.cur is None:
                self._initialize_splits(study)
                self._move_to_next_optimizer(study)
            if self.cur is None: raise ValueError("Please ask before telling.")
            self.cur.tell(trials, study)
        except OutOfBudget:
            if len(trials) > 0: self._move_to_next_optimizer(study)

    def tell_not_asked(self, trial, study):
        if self._no_tell:
            self._no_tell = False
            return
        try:
            if self.cur is None:
                self._initialize_splits(study)
                self._move_to_next_optimizer(study)
            if self.cur is None: raise ValueError("Please ask before telling.")
            self.cur.tell_not_asked(trial, study)
        except OutOfBudget:
            self._move_to_next_optimizer(study)

    def step(self, study):
        self._initialize_splits(study)
        if self.cur is None: self._move_to_next_optimizer(study)
        try: self.cur.step(study) # type:ignore
        except OutOfBudget: self._move_to_next_optimizer(study)


