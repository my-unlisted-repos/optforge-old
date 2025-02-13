import os
from abc import ABC, abstractmethod
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, cast

import numpy as np

from .._types import Numeric, NumericArrayLike
from ..optim.optimizer import Optimizer
from ..paramdict import ParamDict
from ..python_tools import ShutUp, to_valid_fname
from ..registry.optimizers import OPTIMIZERS
from ..study import Study
from ..trial import Trial


class Benchmark(ABC):
    def __init__(self, log_params = True, note:Optional[Any] = None):
        self.study = Study(log_params = log_params)
        self.note = note

    @abstractmethod
    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike": ...

    def run(
        self,
        optimizer: "Optimizer | str",
        max_evals: Optional[int] = None,
        max_steps: Optional[int] = None,
        timeout: Optional[float] = None,
        tol: Optional[float] = None,
        n_jobs: Optional[int] = 1,
        executor: type[Executor] | None = ThreadPoolExecutor,
        catch: tuple[type[Exception], ...] = (),
        progress: bool = True,
        force_batch_mode: bool = False,
        print_results = True,
        disable_prints: bool = False,
        catch_kb_interrupt: bool = True
    ):
        if isinstance(optimizer, str): optimizer = OPTIMIZERS[optimizer]()
        self.optimizer: "Optimizer" = optimizer # type:ignore
        with ShutUp(disable_prints):
            self.study.optimize(
                objective = self.objective,
                optimizer = optimizer,
                max_evals = max_evals,
                max_steps = max_steps,
                timeout = timeout,
                tol = tol,
                progress= progress,
                n_jobs = n_jobs,
                executor=executor,
                catch=catch,
                force_batch_mode=force_batch_mode,
                catch_kb_interrupt = catch_kb_interrupt,
            )
        # self.optimizer = self.study.optimizer
        if print_results:
            print(f'{self.optimizer} achieved best value of {self.study.best_value:8f} in {self.study.current_eval} evals and {self.study.time_passed:.2f} sec')

    def _fname(self):
        now = datetime.now()
        now_str = f'{now.year}.{now.month}.{now.day} {now.hour}-{now.minute}-{now.second}'
        return f'{self.study.best_value:.3f} {self.study.current_eval}e {self.study.time_passed:.1f}s. {now_str}'

    def save(self, dir = 'results', mkdir = True):
        # benchmarks folder
        if mkdir and not os.path.exists(dir): os.mkdir(dir)

        # optimizer folder (./results/optimizer_desc)
        opt_name = to_valid_fname(str(self.optimizer))
        opt_dir = os.path.join(dir, opt_name)
        if not os.path.exists(opt_dir): os.mkdir(opt_dir)

        # filename
        self.study.history.save(os.path.join(opt_dir, f'{self._fname()}.npz'))

    @property
    def best_value(self): return self.study.best_value
    @property
    def best_params(self): return self.study.best_params
    @property
    def history(self): return self.study.history

    def evaluate_params(self, params: dict[str,Any]):
        from ..trial import FixedTrial
        return self.objective(FixedTrial(params))

    def get_minima(self) -> float | np.ndarray | None: ...
    def get_minima_params(self) -> dict[str, Any] | None: ...

    def get_example_params(self) -> dict[str, Any]:
        study = Study()
        study._create_params(self.objective)
        return study.last_trial.params

    def get_num_params(self) -> int:
        study = Study()
        study._create_params(self.objective)
        return study.evaluated_trial.params.numel()

class ExampleSphereBenchmark(Benchmark):
    def objective(self, trial:"Trial") -> "Numeric | NumericArrayLike":
        x = trial.suggest_float('x', -10, 10, init = 9)
        y = trial.suggest_float('y', -10, 10, init = 7)
        return x**2 + y**2
