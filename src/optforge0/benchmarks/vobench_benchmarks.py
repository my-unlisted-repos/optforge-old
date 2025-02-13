import math
from collections.abc import Sequence
from typing import Any, Optional
import os

import numpy as np
import vobench as vb
from vobench.benchmarks import benchmark_base

from .._types import Numeric
from ..study import Study
from ..trial import Trial
from ..python_tools import to_valid_fname, limit_execution_time
from .benchmark import Benchmark

__all__ = [
    "VOBenchmark",
    "VODescentBenchmark",
    "VObenchRunner",
    "VObenchCallback",
]

class VOBenchmark(Benchmark):
    def __init__(self, benchmark:benchmark_base.Benchmark, step = None, scale = None, discrete = False, custom_bounds = False, low=None, high=None, note = None, log_params = True):
        super().__init__(note = note, log_params=log_params)
        self.benchmark = benchmark
        self.step = step
        self.scale = scale
        self.discrete = discrete
        self.low = low
        self.high = high
        self.custom_bounds = custom_bounds

    def objective(self, trial:Trial) -> "Numeric":
        low = self.low if self.custom_bounds else self.benchmark.BOUNDS[0]
        high = self.high if self.custom_bounds else self.benchmark.BOUNDS[1]
        params = trial.suggest_array(
            'params',
            self.benchmark.params.shape,
            low = low,
            high = high,
            step = self.step,
            scale = self.scale,
            discrete_step = self.discrete,
            init = self.benchmark.params
        )
        return self.benchmark.step(params)

    def plot(self, *args, **kwargs):
        self.benchmark.plot(*args, **kwargs)

    def render_video(self, outfile:str, fps = 60, *args, **kwargs): # pylint:disable=W1113
        self.benchmark.render_video(outfile = outfile, fps = fps, *args, **kwargs)

class VObenchRunner:
    def __init__(self, img_dir: Optional[str], video_dir: Optional[str] = None, mkdir = True):
        self.img_dir = img_dir
        self.video_dir = video_dir
        if mkdir:
            if img_dir is not None and not os.path.exists(img_dir): os.mkdir(img_dir)
            if video_dir is not None and not os.path.exists(video_dir): os.mkdir(video_dir)

    def __call__(self, benchmark: VOBenchmark, outdir, **kwargs):
        if hasattr(benchmark, 'study') and benchmark.study.current_eval > 1:
            hard_timeout = kwargs.pop('hard_timeout')
            if hard_timeout is not None: runner = limit_execution_time(hard_timeout)(benchmark.run)
            else: runner = benchmark.run
            runner(**kwargs, print_results=False)
            if outdir is not None: benchmark.save(outdir)

            fname = f'{to_valid_fname(str(benchmark.optimizer))}'
            if len(fname) + len(benchmark._fname()) > 127: fname = fname[:127 - (len(benchmark._fname()) + 5)]+ '...'
            fname += f' {benchmark._fname()}'
            if self.img_dir is not None:
                import matplotlib.pyplot as plt
                benchmark.plot()
                plt.savefig(os.path.join(self.img_dir, f'{fname}.png'))
                plt.close()
            if self.video_dir is not None:
                benchmark.render_video(os.path.join(self.video_dir, f'{fname}.mp4'))

class VObenchCallback:
    def __init__(self, img_dir: Optional[str], video_dir: Optional[str] = None, mkdir = True):
        self.img_dir = img_dir
        self.video_dir = video_dir
        if mkdir:
            if img_dir is not None and not os.path.exists(img_dir): os.mkdir(img_dir)
            if video_dir is not None and not os.path.exists(video_dir): os.mkdir(video_dir)
    def __call__(self, benchmark: VOBenchmark):
        if hasattr(benchmark, 'study') and benchmark.study.current_eval > 1:
            fname = f'{to_valid_fname(str(benchmark.optimizer))}'
            if len(fname) + len(benchmark._fname()) > 127: fname = fname[:127 - (len(benchmark._fname()) + 5)]+ '...'
            fname += f' {benchmark._fname()}'
            if self.img_dir is not None:
                import matplotlib.pyplot as plt
                benchmark.plot()
                plt.savefig(os.path.join(self.img_dir, f'{fname}.png'))
                plt.close()
            if self.video_dir is not None:
                benchmark.render_video(os.path.join(self.video_dir, f'{fname}.mp4'))

class VODescentBenchmark(Benchmark):
    def __init__(self, descent:vb.FunctionDescent2D | vb.ImageDescent, draw_lines = True, note = None, log_params = True):
        super().__init__(note = note, log_params=log_params)
        self.descent = descent
        self.draw_lines = draw_lines

    @classmethod
    def from_func(cls, func, coords, dtype = np.float64, xbounds = None, ybounds = None, xrange = None, yrange = None, clip=True, draw_lines = True, oob_penalty = False, note = None, log_params = True):
        descent = vb.FunctionDescent2D(func, coords, dtype = dtype, xbounds = xbounds, ybounds = ybounds, xrange = xrange, yrange = yrange, clip=clip, oob_penalty=oob_penalty)
        return cls(descent, draw_lines = draw_lines, note = note, log_params = log_params)

    @classmethod
    def from_image(cls, image, coords, dtype = np.float64, clip=False, oob_penalty = True, draw_lines = True, note = None, log_params = True):
        descent = vb.ImageDescent(image, coords, dtype = dtype, clip=clip, oob_penalty=oob_penalty)
        return cls(descent, draw_lines = draw_lines, note = note, log_params = log_params)

    def objective(self, trial:Trial) -> "Numeric":
        xlow = self.descent.xbounds[0] if self.descent.xbounds is not None else None
        xhigh = self.descent.xbounds[1] if self.descent.xbounds is not None else None
        ylow = self.descent.ybounds[0] if self.descent.ybounds is not None else None
        yhigh = self.descent.ybounds[1] if self.descent.ybounds is not None else None
        x = trial.suggest_float('x', low = xlow, high = xhigh, init = self.descent.params[0])
        y = trial.suggest_float('y', low = ylow, high = yhigh, init = self.descent.params[1])
        if self.draw_lines: return self.descent((x, y))
        else: return self.descent.step((x, y))

    def plot(self, *args, **kwargs):
        self.descent.plot(*args, **kwargs)

    def render_video(self, outfile:str, fps = 60, ppf = 1, init_repeats = 1, speedup = 1, speedup_every = 1):
        self.descent.render_video(outfile = outfile, fps = fps, ppf = ppf, init_repeats = init_repeats, speedup = speedup, speedup_every=speedup_every)
