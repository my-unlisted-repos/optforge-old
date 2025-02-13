import os
from collections import UserDict, UserList
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, Optional

import numpy as np

from ._types import f2f_remove1_add1
from .history import History
from .progress import SimpleProgress
from .python_tools import reduce_dim, str_norm


class HistoriesList(UserList[History]):
    def __init__(self, *args, **kwargs):
        """Summary of histories."""
        super().__init__(*args, **kwargs)

    def as_history(self):
        history = History([])
        cur = 0
        for h in self:
            for key in h.data_keys:
                if key not in history:
                    history.data_keys[key] = h.data_keys[key] + cur
                    history.data_values[key] = h.data_values[key]
                else:
                    history.data_keys[key] = np.concatenate([history.data_keys[key], h.data_keys[key] + cur])
                    history.data_values[key] = np.concatenate([history.data_values[key], h.data_values[key]])
            cur += len(h)
        return history

    def add_file(self, file:str):
        self.append(History.from_file(file))

    def add_dir(self, dir:str):
        for file in os.listdir(dir):
            if file.endswith('.npz'):
                self.add_file(os.path.join(dir, file))

    @classmethod
    def from_dir(cls, dir:str) -> "HistoriesList":
        histories = cls()
        histories.add_dir(dir)
        return histories

    def keys(self):
        return set().union(*[i.keys() for i in self])

    def longest(self):
        return max([len(i) for i in self])

    def sorted_lowest(self, key: str) -> "HistoriesList":
        """Returns a sorted list of tuples: `(key, History)`, with 1st element having the lowest `min(value)`."""
        return HistoriesList(sorted(self, key = lambda x: x[key].min()))

    def sorted_highest(self, key: str) -> "HistoriesList":
        """Returns a sorted list of tuples: `(key, min(value))`, with 1st element having the highest `max(value)`."""
        return HistoriesList(sorted(self, key = lambda x: x[key].max(), reverse=True))

    def sorted_best(self, key: str, mode: Literal['lowest', 'highest']) -> "HistoriesList":
        if mode == 'lowest': return self.sorted_lowest(key)
        if mode == 'highest': return self.sorted_highest(key)
        raise ValueError(f"mode must be 'lowest' or 'highest', not {mode}")

    def lowest(self, key) -> History:
        "Returns history with the lowest `min(key)`"
        return self.sorted_lowest(key)[0]

    def highest(self, key) -> History:
        "Returns history with the lowest `min(key)`"
        return self.sorted_highest(key)[0]

    def lowest_value(self, key) -> float:
        "Returns the lowest `min(key)`"
        return float(list(sorted(self, key = lambda x: x[key].min()))[0][key].min())

    def highest_value(self, key) -> float:
        "Returns the highest `max(key)`"
        return float(list(sorted(self, key = lambda x: x[key].max(), reverse=True))[0][key].max())

    def median(self, key, mode: Literal['lowest', 'highest'] = 'lowest') -> History:
        "Returns history with the median `min(key)` or `max(key)"
        histories = self.sorted_best(key, mode)
        return histories[len(histories) // 2]

    def best(self, key, mode: Literal['lowest', 'highest']):
        if mode == 'lowest': return self.lowest(key)
        if mode == 'highest': return self.highest(key)
        raise ValueError(f"mode must be 'lowest' or 'highest', not {mode}")

    def linechart(self, x:str, y:str, color:Any = 'red', xlim = None, ylim = None,ax = None, figsize = None, **kwargs):
        import matplotlib.pyplot as plt

        from .plt_tools import ax_plot_
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        histories = [i.get_valid(x, y) for i in self]
        for xv, yv in histories:
            ax_plot_(ax, xv, yv, xlabel=x, ylabel = y, color=color, alpha=0.4, xlim=xlim, ylim=ylim, **kwargs)

        max_len = max([len(i[0]) for i in histories])
        histories = [i.get_valid_with_length(x, y, length = max_len) for i in self]
        y_vals = [i[1] for i in histories]
        y_low = np.min(y_vals, axis=0)
        y_high = np.max(y_vals, axis=0)
        ax.fill_between(histories[0][0], y_low, y_high, alpha=0.1, step='pre', color=color)

    def plot_median(
        self,
        x:str,
        y:str,
        color:Any="red",
        fill=True,
        alpha=1,
        fill_alpha=0.2,
        xlim=None,
        ylim=None,
        ax=None,
        figsize = None,
        **kwargs
    ):
        import matplotlib.pyplot as plt

        from .plt_tools import ax_plot_
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        ax_plot_(ax, *self.median(y).get_valid(x, y), xlabel=x, ylabel = y, color=color, alpha=alpha, xlim=xlim, ylim=ylim, **kwargs)

        if fill:
            histories = [i.get_valid(x, y) for i in self]
            max_len = max([len(i[0]) for i in histories])
            histories = [i.get_valid_with_length(x, y, length = max_len) for i in self]

            y_vals = [i[1] for i in histories]
            y_low = np.min(y_vals, axis=0)
            y_high = np.max(y_vals, axis=0)
            ax.fill_between(histories[0][0], y_low, y_high, alpha=fill_alpha, step='pre', color=color)


    def plot_convergence(self, color = 'red', xlim = None, ylim = None, plot_all = True, x = 'eval', ax = None, ):
        import matplotlib.pyplot as plt

        from .plt_tools import ax_legend_, ax_scatter_
        if ax is None: ax = plt.gca()

        if plot_all: self.linechart(x, 'best', color = color, xlim = xlim, ylim = ylim, ax = ax)
        else: self.plot_median(x, 'best', color = color, xlim = xlim, ylim = ylim, ax = ax)

        histories = [i.get_valid(x, 'value') for i in self]
        for xv, yv in histories:
            ax_scatter_(ax, xv, yv, xlabel=x, ylabel = 'value', alpha=0.2, xlim=xlim, ylim=ylim, s = 4)

    @f2f_remove1_add1(History.scatter) # type:ignore
    def scatter(self, *args, **kwargs):
        self.as_history().scatter(*args, **kwargs)

    @f2f_remove1_add1(History.scatter3d) # type:ignore
    def scatter3d(self, *args, **kwargs):
        self.as_history().scatter3d(*args, **kwargs)

    @f2f_remove1_add1(History.contour) # type:ignore
    def contour(self, *args, **kwargs):
        self.as_history().contour(*args, **kwargs)

    @f2f_remove1_add1(History.pcolormesh) # type:ignore
    def pcolormesh(self, *args, **kwargs):
        self.as_history().pcolormesh(*args, **kwargs)

    @f2f_remove1_add1(History.surface) # type:ignore
    def surface(self, *args, **kwargs):
        self.as_history().surface(*args, **kwargs)

    @f2f_remove1_add1(History.hist2d) # type:ignore
    def hist2d(self, *args, **kwargs):
        self.as_history().hist2d(*args, **kwargs)


class BenchmarkSummary(UserDict[str, HistoriesList]):

    def add_dir(self, dir: str, name:str):
        self[name] = HistoriesList.from_dir(dir)

    def add_benchmark_dir(self, dir:str, filt:Optional[Callable[[str], bool]] = None, progress=True):
        for i in SimpleProgress(os.listdir(dir), enable=progress):
            if os.path.isdir(os.path.join(dir, i)):
                if filt is None or filt(i):
                    self.add_dir(os.path.join(dir, i), i)

    @classmethod
    def from_benchmark_dir(cls, dir:str, filt:Optional[Callable[[str], bool]] = None, progress=True):
        s = cls()
        s.add_benchmark_dir(dir, filt = filt, progress=progress)
        if len(s) == 0: raise ValueError(f"No benchmarks found in {dir}")
        return s

    def filtered(self, filt:Callable[[str], bool]):
        return BenchmarkSummary({k:v for k,v in self.items() if filt(k)})

    def as_history_list(self, filt: Optional[Callable[[str], bool]] = None) -> HistoriesList:
        histories = [v for k,v in self.items() if filt is None or filt(k)]
        return HistoriesList(reduce_dim(histories))

    def as_history(self, filt: Optional[Callable[[str], bool]] = None) -> "History":
        return self.as_history_list(filt = filt).as_history()

    def history_keys(self):
        return set(reduce_dim(i.keys() for i in self.values()))

    def sort(self, key:str = 'value'):
        self.data = {k:v for k,v in sorted(self.items(), key = lambda x: (x[1].median(key).min(key), x[1].lowest(key).min(key)))}

    def sorted(self, key:str = 'value'):
        return BenchmarkSummary({k:v for k,v in sorted(self.items(), key = lambda x: (x[1].median(key).min(key), x[1].lowest(key).min(key)))})

    def summary(self, key:str = 'value'):
        for k,v in self.sorted(key).items():
            v_sorted = v.sorted_lowest(key)
            median = v.median(key)
            print(f"{median.min(key):.3f} | {v_sorted[0].min(key):.3f} - {v_sorted[-1].min(key):.3f}: {k}")

    def plot_convergence(
        self,
        filt: Optional[Callable[[str], bool]] = None,
        xlim=None,
        ylim=None,
        plot_all=False,
        scatter=False,
        x="eval",
        legend = True,
        colors:Optional[Sequence[str]] = (
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "cyan",
            "black",
            "magenta",
            "lime",
            "pink",
            "gray",
            "yellow",
            "brown",
            "teal",
            "olive",
            "navy",
            "maroon",
            "fuchsia",
        ),
        loc='lower center',
        frameon=True,
        framealpha=0.5,
        edgecolor="gray",
        facecolor="white",
        markerscale=1.5,
        bbox_to_anchor = (0.5,-0.4),
        ncol = 2,
        figsize = (14, 6),
        ax=None,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        from .plt_tools import ax_legend_, ax_scatter_
        if ax is None: fig, ax = plt.subplots(figsize=figsize, layout='constrained')

        d = self if filt is None else self.filtered(filt)

        for (name, hlist), color in zip(d.items(), colors if colors is not None else [None for _ in d]):
            if plot_all: hlist.linechart(x, 'best', color = color, xlim = xlim, ylim = ylim, ax = ax, label = name, **kwargs)
            else: hlist.plot_median(x, 'best', color = color, xlim = xlim, ylim = ylim, ax = ax, label = name, fill_alpha=0.1, **kwargs)

            if scatter:
                histories = [i.get_valid(x, 'value') for i in hlist]
                for xv, yv in histories:
                    ax_scatter_(ax, xv, yv, c = color, xlabel=x, ylabel = 'value', alpha=0.2, xlim=xlim, ylim=ylim, s = 4)

        if legend:
            ax_legend_(
                ax,
                loc=loc,
                frameon=frameon,
                framealpha=framealpha,
                edgecolor=edgecolor,
                facecolor=facecolor,
                markerscale=markerscale,
                bbox_to_anchor = bbox_to_anchor,
                ncol = ncol,
            )

    def compare(
        self,
        *items,
        relaxed=True,
        xlim=None,
        ylim=None,
        plot_all=False,
        scatter=False,
        x="eval",
        legend = True,
        colors:Optional[Sequence[str]] = (
            "red",
            "green",
            "blue",
            "orange",
            "purple",
            "cyan",
            "black",
            "magenta",
            "lime",
            "pink",
            "gray",
        ),
        loc='lower center',
        frameon=True,
        framealpha=0.5,
        edgecolor="gray",
        facecolor="white",
        markerscale=1.5,
        bbox_to_anchor = (0.5,-0.25),
        ncol = 2,
        figsize = (14, 6),
        ax=None,
        **kwargs,
        ):
        if relaxed:
            keys = []
            for i in items:
                if i not in self: keys.extend([k for k in self.keys() if str_norm(i) in str_norm(k)])
                else: keys.append(i)
        else: keys = items
        self.plot_convergence(
            filt=lambda x: x in keys,
            xlim=xlim,
            ylim=ylim,
            plot_all=plot_all,
            scatter=scatter,
            x=x,
            ax=ax,
            legend=legend,
            colors=colors,
            loc=loc,
            frameon=frameon,
            framealpha=framealpha,
            edgecolor=edgecolor,
            facecolor=facecolor,
            markerscale=markerscale,
            bbox_to_anchor = bbox_to_anchor,
            ncol = ncol,
            figsize = figsize,
            **kwargs,
        )