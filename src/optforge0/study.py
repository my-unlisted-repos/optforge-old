import itertools
import os
import time
from collections.abc import Callable, Generator, Iterable, Sequence
from concurrent.futures import (FIRST_COMPLETED, Executor, Future,
                                ThreadPoolExecutor, wait)
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import numpy as np

from ._types import NumericArrayLike, Numeric
from .constraint_handler import (_default_constraint_apply,
                                 _default_constraint_reduce)
from .history import History
from .optim.optimizer import Optimizer
from .optim.debug import _DefaultOptimizer
from .paramdict import ParamDict
from .pareto import is_pareto_front
from .python_tools import reduce_dim
from .registry.optimizers import OPTIMIZERS
from .trial.finished_trial import FinishedTrial, _init_trial
from .trial.trial import Trial, VectorizedObjectiveTrialPack
from .pruners.base_pruner import NoopPruner, Pruner

inf = float('inf')

__all__ = [
    "Study",
    "EndStudy"
]

class EndStudy(Exception):
    """An exception that can be raised to end `study.optimize`."""


nan = float('nan')

def _call(x): return x()
_formatter = {'float': lambda x: f'{x:.4f}'}


def _oneplusprod(x:np.ndarray) -> float:
    return float(np.prod(x + 1))
class Study:
    """A study."""
    def __init__(
        self,
        log_params: bool = True,
        mo_indicator: "Callable[[np.ndarray], Numeric]" = _oneplusprod,
        soft_handler: Callable[[np.ndarray], float] = _default_constraint_reduce,
        hard_handler: Callable[[np.ndarray], float] = _default_constraint_reduce,
        applier: Callable[[float | np.ndarray, float], float | np.ndarray] = _default_constraint_apply,
        note: Any = None,
    ):
        self.note :Any = note
        """Any additional information about the study."""
        self.log_params: bool = log_params
        """Wheher to log parameters on each evaluation. Defaults to True"""
        self.mo_indicator: Callable[[np.ndarray], Numeric] = mo_indicator
        """A multi-objective indicator callable that accepts a 1D numpy.ndarray with each objective value and returns indicator scalar. Defaults to np.mean"""
        self.soft_handler: Callable[[np.ndarray], float] = soft_handler
        """Function that reduces multiple soft constraint violations to a single value."""
        self.hard_handler: Callable[[np.ndarray], float] = hard_handler
        """Function that reduces multiple hard constraint violations to a single value."""
        self.applier: Callable[[float | np.ndarray, float], float | np.ndarray] = applier
        """Function that reduces multiple soft constraint violations to a single value."""

        self.trials: list[FinishedTrial] = []
        """List of all finished trials in order of their completion."""
        self.best_trial: FinishedTrial = _init_trial
        """Finished trial that achieved the lowest value. In case of multiobjective optimization this is the trial within the pareto front with the lowest indicator value."""
        self.last_trial: FinishedTrial = _init_trial
        """Last finished trial."""
        self.best_indicator_trial: FinishedTrial = _init_trial
        """Trial that achieved best indicator value, unlike `best_trial`, not guaranteed to be in pareto front."""
        self.pareto_front_trials: list[FinishedTrial] = []
        """A list of pareto front trials."""

        self.current_eval:int = 0
        """Number of evaluations of the objective function performed by this study."""
        self.current_step:int = 0
        """Number of optimizer steps performed by this study. Some optimizers do multiple evaluations per step."""

        # start time gets recorded by calling time.time()
        # when `step` or `ask` are called and this is None,
        # or always on `optimize` start.
        self.start_time:float = nan
        """Start time of the study, in seconds from the epoch."""
        self.end_time:float = nan
        """End time of the study, in seconds from the epoch."""
        self.time_passed: float = 0
        """Time passed since the study started, in seconds."""

        self.storage = {}
        """Global storage that optimizers can use to store global stuff."""

        # ...
        self._enable_print_progress:bool = False
        self._last_print_time = 0

        # stop conditions
        self.max_evals: int | None = None
        """Maximum number of evaluations. If this is exeeded, and EndStudy is raised and catched by `optimize` loop. Defaults to None"""
        self.max_steps: int | None = None
        """Maximum number of optimizer steps. If this is exeeded, and EndStudy is raised and catched by `optimize` loop. Defaults to None"""
        self.timeout: float | None = None
        """Maximum time to run the study in seconds. If this is exeeded, and EndStudy is raised and catched by `optimize` loop. Defaults to None"""
        self.tol: float | None = None
        """Tolerance for convergence. If the best value becomes lower than this, an EndStudy is raised and catched by `optimize` loop. Defaults to None"""
        self.callbacks: list[Callable[[Study, Trial, FinishedTrial], None]] = []
        """List of callbacks to be called after each evaluation. Callbacks are called with two arguments: study and finished_trial."""

        self._current_eval_logs: dict[Any, Any] = {}

        self.improved = False
        """Whether last evaluation improved the best value."""

        self._history: Optional[History] = None

        self.optimizer = _DefaultOptimizer()

        self.is_viable = False

        self.pruner: Pruner | None = None

    def set_stop_conditions(self, max_evals: Optional[int] = None, max_steps: Optional[int] = None, timeout: Optional[float] = None, tol: Optional[float] = None):
        """Set or reset the stop conditions for the study.

        :param max_evals: Maximum number of evaluations. If this is exeeded, and EndStudy is raised and catched by `optimize` loop. Defaults to None
        :param max_steps: Maximum number of optimizer steps. If this is exeeded, and EndStudy is raised and catched by `optimize` loop. Defaults to None
        :param timeout: Maximum time to run the study in seconds. If this is exeeded, and EndStudy is raised and catched by `optimize` loop. Defaults to None
        :param tol: Tolerance for convergence. If the best value becomes lower than this, an EndStudy is raised and catched by `optimize` loop. Defaults to None
        """
        self.max_evals = max_evals
        self.max_steps = max_steps
        self.timeout = timeout
        self.tol = tol

    @property
    def best_value(self):
        """Best value so far. In case of multi-objective optimization returns the scalar indicator.
        If you need to access multi-objective values, use `best_trial.value` for best trial, or `pareto_front_values` for all pareto front values."""
        return self.best_trial.scalar_value

    def get_best_value(self, soft_penalty=True, scalar=True):
        return self.best_trial.objective_value.get(
            soft_penalty=soft_penalty,
            hard_penalty=True,
            param_penalty=False,
            scalar=scalar,
        )

    @property
    def pareto_front_values(self) -> list[np.ndarray]:
        """List of pareto-front values."""
        return [i.value for i in self.pareto_front_trials] # type:ignore
    @property
    def best_params(self):
        """Best params so far."""
        return self.best_trial.params
    @property
    def pareto_front_params(self) -> list[dict[str, Any]]:
        """List of pareto-front params."""
        return [i.params for i in self.pareto_front_trials]

    def _create_params(self, objective: "Callable[[Trial], Numeric | NumericArrayLike]",):
        """Creates and populates the initial paramdict by evaluating objective with initially suggesting params"""
        if self.start_time is None: self.start_time = time.time()
        self.objective = objective
        # create a trial with empty paramdict
        trial = Trial(paramdict = ParamDict(), study = self, asked=False)
        if self.start_time is None: self.start_time = time.time()

        # Calling the trial evaluates objective with initially suggested parameters and populates paramdict with them.
        # It also creates a finished trial so we submit it.
        trial()


        self._submit_evaluated_trial(trial)
        return trial

    def _print_progress(self):
        t = ''
        if self.max_evals is not None: t += f' | evals: {self.current_eval}/{self.max_evals}'
        if self.max_steps is not None: t += f' | steps: {self.current_step}/{self.max_steps}'
        if self.timeout is not None: t += f' | time: {self.time_passed:.2f}s/{self.timeout}s'
        value = self.best_trial.value
        if isinstance(value, np.ndarray):
            if value.size < 8:
                t += f' | best value: {np.array2string(value, formatter=_formatter)}' # type:ignore
            else: t += f' | best indicator: {self.best_trial.scalar_value:.4f}'
        else:
            if value < 1e4: t += f' | best value: {value:.4f}'
            else: t += f' | best value: {value:.2e}'
        if self.tol is not None: t += f'/{self.tol}'
        print(t[3:], end = '                            \r')

    def log(self, key:Any, value:Any):
        self._current_eval_logs[key] = value

    @property
    def history(self) -> History:
        """History of finished trials, can be accessed like a dictionary and can do plotting."""
        if self._history is None or len(self._history) != self.current_eval:  self._history = History(self.trials)
        return self._history

    def _check_stop_conditions(self):
        if self.max_evals is not None and self.current_eval >= self.max_evals: return True
        if self.max_steps is not None and self.current_step >= self.max_steps: return True
        if self.timeout is not None and self.time_passed >= self.timeout: return True
        if self.tol is not None and np.all(self.best_trial.scalar_value <= self.tol): return True

    def _check_next_stop_conditions(self):
        """Checks whether the next trial will raise EndStudy. This is used to `break` after the next trial is created
        instead of force-stopping by raising exceptions, which sometimes leads to verbose warnings from some optimization libraries."""
        if self.max_evals is not None and self.current_eval + 1 >= self.max_evals: return True
        if self.max_steps is not None and self.current_step + 1 >= self.max_steps: return True
        # here we check if we hit 99% of timeout
        if self.timeout is not None and self.time_passed >= self.timeout * 0.99: return True

    def _last_step(self, objective, optimizer):
        if self.max_evals is not None: self.max_evals += 1
        if self.max_steps is not None: self.max_steps += 1
        self.step(objective, optimizer, )
        if self.max_evals is not None: self.max_evals -= 1
        if self.max_steps is not None: self.max_steps -= 1

    def _submit_evaluated_trial(self, evaluated_trial:Trial):
        """Submit a finished trial to the study, update best value and pareto front."""
        self.evaluated_trial = evaluated_trial

        finished_trial = evaluated_trial.finished_trial
        self.trials.append(finished_trial)

        self.improved = False

        if finished_trial.is_viable or not self.is_viable:

            # single objective
            if isinstance(finished_trial.value, float):
                if finished_trial.objective_value.opt_value < self.best_trial.objective_value.opt_value: self.improved = True

            # multi-objective
            else:
                if finished_trial.scalar_value < self.best_indicator_trial.scalar_value:
                    self.best_indicator_trial = finished_trial

                candidates = [finished_trial] + self.pareto_front_trials
                fronts = is_pareto_front([i.value for i in candidates]) # returns a list of bools # type:ignore
                if fronts[0]: # means the new trial is pareto front
                    # if new trial is not pareto front, self.pareto_front_trials is already guaranteed to be pareto front
                    self.pareto_front_evaluations = [c for c, f in zip(candidates, fronts) if f]
                    # also if indicator value is better, this is considered to be the best trial
                    # but that won't trigger if new candidate is not in pareto front
                    # which is how this is different from `best_indicator_trial`
                    if finished_trial.objective_value.opt_scalar_value < self.best_trial.objective_value.opt_scalar_value:
                        self.improved = True

            if finished_trial.is_viable and not self.is_viable: self.improved = True
            self.is_viable = finished_trial.is_viable

        if self.improved:
            self.best_trial = finished_trial
            if self.optimizer.STORE_PARAMDICTS == 'best': self.best_trial.paramdict = evaluated_trial.params.copy()


        self.last_trial = finished_trial

        self.end_time = finished_trial.end_time
        self.time_passed = self.end_time - self.start_time

        finished_trial.current_eval = self.current_eval
        finished_trial.best_value = self.best_trial.value
        finished_trial.time_passed = self.time_passed
        finished_trial.improved = self.improved
        finished_trial.logs.update(self._current_eval_logs)
        self._current_eval_logs = {}

        evaluated_trial.improved = self.improved
        evaluated_trial.best_value = self.best_trial.objective_value.opt_value
        evaluated_trial.best_scalar_value = self.best_trial.objective_value.opt_scalar_value

        if self.pruner is not None: self.pruner.current_eval = self.current_eval

        self.optimizer._scheduler_step(self)

        # callbacks
        if self.callbacks is not None:
            for callback in self.callbacks:
                callback(self, evaluated_trial, finished_trial)
        if self.optimizer.callbacks is not None:
            for callback in self.optimizer.callbacks:
                callback(self, evaluated_trial, finished_trial)

        # increment current_eval after executing callbacks
        # so that they get the right eval
        self.current_eval += 1

        # stop conditions
        if self._check_stop_conditions(): raise EndStudy()

        # print progress
        if self._enable_print_progress:
            # print every second so that no slowdown happens due to fast prints
            if self.end_time - self._last_print_time > 1:
                self._last_print_time = self.end_time
                self._print_progress()

    def set_pruner(self, pruner: Pruner | None):
        self.pruner = pruner

    def evaluate_params(self, params: ParamDict, return_scalar = False) -> float | np.ndarray:
        if self.start_time is None: self.start_time = time.time()

        # reset used_params so that they will get repopulated on evaluation
        self.optimizer.params.used_params = set()

        trial = self.optimizer.trial_cls(paramdict = params, study = self, asked=False)

        # evaluate trial with current parameters
        # also this can add new parameters to paramdict if they are conditional.
        trial()
        self._submit_evaluated_trial(trial)
        if return_scalar: return trial.scalar_value
        return trial.value

    def evaluate(self, return_scalar = False) -> float | np.ndarray:
        return self.evaluate_params(self.optimizer.params, return_scalar=return_scalar)

    def evaluate_return_scalar(self) -> float:
        return self.evaluate_params(self.optimizer.params, return_scalar=True) # type:ignore

    def evaluate_vec(self, vec: np.ndarray, slices: dict[str, slice], return_scalar = False):
        self.optimizer.params.vec_to_params_(vec, slices = slices)
        return self.evaluate(return_scalar = return_scalar)

    def evaluate_array_dict(self, d: dict[str, np.ndarray], normalized = True, return_scalar = False):
        self.optimizer.params.array_dict_to_params_(d, normalized=normalized)
        return self.evaluate(return_scalar = return_scalar)

    def evaluate_scalar_dict(self, d: dict[str, Numeric], normalized = True, return_scalar = False):
        self.optimizer.params.scalar_dict_to_params_(d, normalized = normalized)
        return self.evaluate(return_scalar = return_scalar)

    def __call__(self, return_scalar = False):
        """Evaluate the objective with current parameters"""
        return self.evaluate(return_scalar = return_scalar)

    def _initialize(
        self,
        objective: "Callable[..., Numeric | NumericArrayLike]",
        optimizer: "Optimizer | str",
    ) -> None:
        if self.start_time is None: self.start_time = time.time()
        if isinstance(optimizer, str): optimizer = OPTIMIZERS[optimizer]()
        self.optimizer: "Optimizer" = optimizer # type:ignore
        self.objective = objective

        # if optimizer params are empty, create and set them
        if len(self.optimizer.params) == 0:
            trial = self._create_params(objective)
            self.optimizer.set_params(trial.params)
            self.optimizer.tell_not_asked(trial, self)

    def ask(
        self,
        objective: "Callable[[Trial], Numeric | NumericArrayLike]",
        optimizer: "Optimizer",
        step_fallback: bool = False,
    ) -> Generator[Trial]:
        """_summary_

        :param objective: _description_
        :param optimizer: _description_
        :param step_fallback: _description_, defaults to False
        :raises ValueError: _description_
        :yield: _description_
        """
        self._initialize(objective, optimizer)

        # if optimizer doesn't support ask interface,
        # a step will be made instead for API consistency
        if self.optimizer.SUPPORTS_ASK: yield from self.optimizer._internal_ask(self)
        elif step_fallback: self.optimizer._internal_step(self)
        else: raise ValueError(f"{optimizer.__class__.__name__} doesn't support ask and tell interface.")


    def tell(
        self,
        objective: "Callable[[Trial], Numeric | NumericArrayLike]",
        optimizer: "Optimizer",
        trials: list[Trial],
    ):
        """_summary_

        :param objective: _description_
        :param optimizer: _description_
        :param trials: _description_
        :return: _description_
        """
        self._initialize(objective, optimizer)
        for t in trials: self._submit_evaluated_trial(t)
        self.optimizer._internal_tell(trials, self)
        self.current_step += 1

    def tell_not_asked(
        self,
        objective: "Callable[[Trial], Numeric | NumericArrayLike]",
        optimizer: "Optimizer",
        params: dict[str, Any],
        value: float | np.ndarray
    ):
        """_summary_

        :param objective: _description_
        :param optimizer: _description_
        :param trials: _description_
        :return: _description_
        """
        self._initialize(objective, optimizer)
        # this method also submits the evaluated trial
        self.optimizer._tell_not_asked_orig_params(params, value, self)

    def step(
        self,
        objective: "Callable[[Trial], Numeric | NumericArrayLike]",
        optimizer: "Optimizer",
    ):
        """_summary_

        :param objective: _description_
        :param optimizer: _description_
        :return: _description_
        """
        self._initialize(objective, optimizer)

        # make a step with the optimizer
        # optimizer will use call this study (use __call__), potentially multiple times.
        # and will change the paramdict
        value = self.optimizer._internal_step(self)
        self.current_step += 1
        return value

    def stop(self):
        """_summary_

        :raises EndStudy: _description_
        """
        raise EndStudy()


    def _optimize_single_worker(
        self,
        catch: tuple[type[Exception], ...],
    ):
        # no `for self.current_step` there because calling `self.step` increments `current_step`, as you can call it manually.
        while True:
            try:
                # we try to break without raising EndStudy to avoid force-stopping whenever possible
                # as there are a few libraries that use threading and
                # are very verbose when you force-stop during an optimization step
                # if self.current_eval - 1 = self.max_evals, next step will raise EndStudy
                if self._check_next_stop_conditions():
                    # do the last step and break
                    # last step increments max_evals and max_steps by 1 so that it doesn't raise EndStudy
                    # because we stop by breaking out of the loop,  unless optimizer does more than 1 evaluation.
                    self._last_step(self.objective, self.optimizer)
                    break

                self.step(self.objective, self.optimizer)
            except catch: pass

    def _optimize_multiple_workers_batched(
        self,
        catch: tuple[type[Exception], ...],
        executor: type[Executor],
        n_jobs: int,
    ):
        if not self.optimizer.SUPPORTS_ASK: raise ValueError(f"{self.optimizer.__class__.__name__} doesn't support multiple workers")

        with executor(n_jobs) as ex:# type:ignore
            for self.current_step in itertools.count():

                if self._check_next_stop_conditions():
                    self._last_step(self.objective, self.optimizer)
                    break

                try:
                    trials = self.optimizer._internal_ask(self)
                    trials: "Iterable[Trial]" = list(ex.map(_call, trials))
                    for trial in trials: self._submit_evaluated_trial(trial)
                    self.optimizer._internal_tell(trials, self)
                except catch: pass

    def _optimize_multiple_workers_continuous(
        self,
        catch: tuple[type[Exception], ...],
        executor: type[Executor],
        n_jobs: int,

    ):
        if not self.optimizer.SUPPORTS_ASK: raise ValueError(f"{self.optimizer.__class__.__name__} doesn't support multiple workers")

        multi_ask = self.optimizer.SUPPORTS_MULTIPLE_ASKS

        with executor(n_jobs) as ex: # type:ignore
            trial_packs: "dict[int, list[Trial]]" = {}
            futures: set[Future] = set()
            for self.current_step in itertools.count():
                try:
                    if len(futures) >= n_jobs or (len(futures) > 0 and not multi_ask):
                        completed, futures = wait(futures, return_when=FIRST_COMPLETED)
                        for f in completed: f.result()

                        for i, pack in trial_packs.copy().items():

                            # if all evaluations in the pack have been completed, tell them to the optimizer
                            # all are guaranteed to be completed if not multi_ask
                            if all([e.evaluated for e in pack]):
                                del trial_packs[i]

                                # update self
                                for trial in pack:
                                    self._submit_evaluated_trial(trial)

                                # this can reduce multi-objective values
                                # so it needs to be called after study._update.
                                self.optimizer._internal_tell(pack, self)

                        if self._check_next_stop_conditions():
                            self._last_step(self.objective, self.optimizer)
                            break

                    if self.current_step in trial_packs:
                        raise ValueError(f'{self.current_step = } is already in {trial_packs = }')

                    current_step_trials = trial_packs[self.current_step] = []
                    for trial in self.optimizer._internal_ask(self):
                        current_step_trials.append(trial)
                        futures.add(
                            # submit an Evaluate
                            ex.submit(trial)
                        )
                except catch: pass

    def _optimize_vectorized(
        self,
        catch: tuple[type[Exception], ...],
        num_asks: int,
    ):
        if not self.optimizer.SUPPORTS_ASK: raise ValueError(f"{self.optimizer.__class__.__name__} doesn't support ask interface, which is required for vectorized evaluation.")
        if not self.optimizer.SUPPORTS_MULTIPLE_ASKS: num_asks = 1
        for self.current_step in itertools.count():
            try:
                trial_packs = [list(self.optimizer._internal_ask(self)) for _ in range(num_asks)]
                trial_pack_lengths = [len(pack) for pack in trial_packs]
                vectorized = VectorizedObjectiveTrialPack(reduce_dim(trial_packs))
                trials = vectorized()
                for t in trials: self._submit_evaluated_trial(t)
                cur = 0
                for l in trial_pack_lengths:
                    self.optimizer._internal_tell(trials[cur:cur+l], self)
                    cur += l
            except catch: pass

    def optimize(
        self,
        objective: "Callable[[Trial], Numeric | NumericArrayLike]",
        optimizer: "Optimizer | str",
        max_evals: "Optional[int]" = None,
        max_steps: "Optional[int]" = None,
        timeout: "Optional[float]" = None,
        tol: "Optional[float]" = None,
        n_jobs: "Optional[int]" = 1,
        executor: Optional[type[Executor]] = ThreadPoolExecutor,
        catch: tuple[type[Exception], ...] = (),
        progress: bool = True,
        force_batch_mode:bool = False,
        vectorized: Literal[False] | int = False,
        catch_kb_interrupt: bool = True,
        seed: Optional[int] = None,
    ):
        """_summary_

        :param objective: _description_
        :param optimizer: _description_
        :param max_evals: _description_, defaults to None
        :param max_steps: _description_, defaults to None
        :param timeout: _description_, defaults to None
        :param tol: _description_, defaults to None
        :param n_jobs: _description_, defaults to 1
        :param executor: _description_, defaults to ThreadPoolExecutor
        :param catch: _description_, defaults to ()
        :param progress: _description_, defaults to True
        :param force_batch_mode: _description_, defaults to False
        :return: _description_
        """
        self.start_time = time.time()
        if isinstance(optimizer, str): optimizer = OPTIMIZERS[optimizer]()
        self.optimizer: "Optimizer" = optimizer # type:ignore
        if seed is not None: self.optimizer.set_seed(seed)

        self.objective = objective
        self.max_evals = max_evals; self.max_steps = max_steps; self.timeout = timeout; self.tol = tol
        if n_jobs is None: n_jobs = os.cpu_count() or 1

        self._enable_print_progress = progress
        self._last_print_time = self.start_time

        kb_interrupt = KeyboardInterrupt if catch_kb_interrupt else ()

        try:
            # if optimizer params are empty, create and set them
            if self.optimizer.budget is None: self.optimizer.set_budget(self.max_evals)
            if len(self.optimizer.params) == 0:
                trial = self._create_params(objective)
                self.optimizer.set_params(trial.params)
                # also suggest that trial to the optimizer, otherwise it might start from randomized params
                self.optimizer.tell_not_asked(trial, self)

            if vectorized and self.optimizer.SUPPORTS_ASK:
                self._optimize_vectorized(catch = catch, num_asks = vectorized)
            elif executor is None or n_jobs <= 1 or not self.optimizer.SUPPORTS_ASK:
                self._optimize_single_worker(catch = catch)
            elif self.optimizer.REQUIRES_BATCH_MODE or force_batch_mode:
                self._optimize_multiple_workers_batched(catch = catch, executor = executor, n_jobs = n_jobs,)
            else:
                self._optimize_multiple_workers_continuous(catch = catch, executor = executor, n_jobs = n_jobs,)

        except EndStudy:
            pass
        except kb_interrupt:
            print('KeyboardInterrupt, stopping.')


        if progress:
            self._print_progress()
            print()
            print(f'Job finished in {self.time_passed:.1f}s., did {self.current_eval} evaluations.')

        self._enable_print_progress = False