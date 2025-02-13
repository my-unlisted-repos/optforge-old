from collections.abc import Collection, Sequence, Callable
from typing import Optional
import os
from datetime import datetime
from ..python_tools import ShutUp, limit_execution_time_raise
# ...
from ..registry.optimizers import OPTIMIZERS
from ..registry.groups import GROUPS
from .benchmark import Benchmark

# ...


def default_runner(benchmark:Benchmark, outdir, **kwargs):
    hard_timeout = kwargs.pop('hard_timeout')
    if hard_timeout is not None: runner = limit_execution_time_raise(hard_timeout)(benchmark.run)
    else: runner = benchmark.run
    runner(**kwargs, print_results=False, catch_kb_interrupt = False)
    if outdir is not None: benchmark.save(outdir)


def test_one_optimizer(
    optimizer,
    benchmark_constructor: Callable[[], Benchmark],
    max_evals: Optional[int],
    timeout: Optional[float],
    hard_timeout: Optional[float] = None,
    disable_prints:bool = True,
    outdir: Optional[str] = 'results',
    runs_per_optimizer: int = 1,
    runner:Callable = default_runner,
    cbs: Sequence[Callable[[Benchmark], None]] = (),
    print_progress: bool = True,
    print_failed: bool = True,
    mkdir = True,
    name: Optional[str] = None,
    catch = (Exception,),
):
    if mkdir and outdir is not None and not os.path.exists(outdir): os.mkdir(outdir)
    if isinstance(optimizer, str): optimizer = OPTIMIZERS[optimizer]
    if name is None: name = optimizer.name # type:ignore
    hit_hard_timeout = False
    for _ in range(runs_per_optimizer):
        if print_progress: print(f'{datetime.now()} - {name}', end = '\r')
        with ShutUp(disable_prints):
            exceptions = []
            benchmark: Benchmark = benchmark_constructor()
            try:
                opt = optimizer()
            except Exception as e:
                print(f"Can't instantiate optimizer {name}: {e!r}")
                continue
            try:
                runner(benchmark, outdir=outdir, optimizer = opt, max_evals=max_evals, timeout=timeout,hard_timeout=hard_timeout, )
            except TimeoutError as e:
                exceptions.append(e)
                hit_hard_timeout = True
            except catch as e:
                exceptions.append(e)
            finally:
                if benchmark.study.current_eval > 0:
                    for cb in cbs: cb(benchmark)
        if print_failed and len(exceptions) > 0: print(f'FAILED: {datetime.now()} - {name} failed with {exceptions[-1]!r}')
        if print_progress: print(f'{datetime.now()} - {name} achieved {benchmark.best_value:.3f} in {benchmark.study.current_eval} evals, {benchmark.study.time_passed:.2f} seconds.', )
        if hit_hard_timeout: break

def test_all_optimizers(
    benchmark_constructor: Callable[[], Benchmark],
    max_evals: Optional[int],
    timeout: Optional[float],
    hard_timeout: Optional[float] = None,
    groups:str | Sequence[str] = GROUPS.MAIN,
    dims: Optional[int] = None,
    blacklist: Sequence[str] | Collection[str] = (),
    disable_prints:bool = True,
    outdir: Optional[str] = 'results',
    runs_per_optimizer: int = 1,
    cbs: Sequence[Callable[[Benchmark], None]] = (),
    runner:Callable = default_runner,
    print_progress: bool = True,
    print_failed: bool = True,
    mkdir = True,
    start_from: Optional[str] = None,
    skip_start_from: bool = False,
    filt: Optional[Callable[[str], bool]] = None,
    catch = (Exception,),
):
    if isinstance(groups, str): groups = [groups]
    if mkdir and outdir is not None and not os.path.exists(outdir): os.mkdir(outdir)
    for name, optimizer in OPTIMIZERS.sorted_items(groups):
        if start_from is not None:
            if name < start_from: continue
            elif skip_start_from:
                skip_start_from = False
                continue

        if name in blacklist: continue
        if dims is not None and optimizer.maxdims is not None and dims > optimizer.maxdims: continue
        if filt is not None and not filt(name): continue

        test_one_optimizer(
            name = name,
            optimizer = optimizer,
            benchmark_constructor = benchmark_constructor,
            max_evals = max_evals,
            timeout = timeout,
            hard_timeout=hard_timeout,
            disable_prints = disable_prints,
            outdir = outdir,
            runs_per_optimizer = runs_per_optimizer,
            runner = runner,
            cbs = cbs,
            print_progress = print_progress,
            print_failed = print_failed,
            mkdir = mkdir,
            catch = catch,
        )