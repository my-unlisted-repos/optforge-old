from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, Self

import numpy as np

from .trial import FinishedTrial
from ._types import f2f_remove3_add1

nan = float('nan')

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from mpl_toolkits.mplot3d.axes3d import Axes3D

def _randbinary_exact_proportion(p, size):
    """P is exact proportion of ones"""
    size = np.prod(size)
    n = int(round(p * size))
    vals = np.zeros(size, dtype=int)
    vals[:n] = 1
    np.random.shuffle(vals)
    return vals.reshape(size)

class History:
    def __init__(self, trials: list[FinishedTrial]):
        keys = (
            set(f"params.{key}" for trial in trials for key in trial.params)
            .union(set(f"logs.{key}" for trial in trials for key in trial.logs))
            .union(
                {
                    "value",
                    "best_value",
                    "scalar_value",
                    "original_value",
                    "current_step",
                    "current_eval",
                    "time_passed",
                    "soft_violations",
                    "hard_violations",
                    "total_soft_violation",
                    "total_hard_violation",
                    "is_viable",
                    "improved",
                }
            )
        )

        self.aliases = {
            "best": "best_value",
            "original": "original_value",
            "scalar": "scalar_value",
            "step": "current_step",
            "eval": "current_eval",
            "time": "time_passed",
            "soft": "total_soft_violation",
            "hard": "total_hard_violation",
        }

        self.data_keys: dict[str, np.ndarray] = {}
        self.data_values: dict[str, np.ndarray] = {}

        dictkeys = ('params.', 'logs.', 'soft_violations.', 'hard_violations.')
        for key in keys:
            if key.startswith(dictkeys):
                attr, k = key.split('.', maxsplit=1)
                trials_with_key = [trial for trial in trials if k in getattr(trial, attr)]
                self.data_keys[key] = np.array([trial.current_eval for trial in trials_with_key])
                self.data_values[key] = np.array([getattr(trial, attr)[k] for trial in trials_with_key])

            else:
                trials = [trial for trial in trials if hasattr(trial, 'current_eval')]
                self.data_keys[key] = np.array([trial.current_eval for trial in trials])
                self.data_values[key] = np.array([getattr(trial, key) for trial in trials])

        self._randmasks = {}

    def keys(self):
        return self.data_keys.keys()

    def __len__(self):
        return int(self.data_keys['current_eval'][-1] + 1)

    def save(self, filepath:str):
        keys = {f'{k}.__key__':v for k,v in self.data_keys.items()}
        values = {f'{k}.__value__':v for k,v in self.data_values.items()}
        np.savez_compressed(filepath, **keys, **values)

    def load(self, filepath:str):
        arrays = np.load(filepath, allow_pickle=True)
        self.data_keys = {k.replace('.__key__', ''):v for k,v in arrays.items() if k.endswith('.__key__')}
        self.data_values = {k.replace('.__value__', ''):v for k,v in arrays.items() if k.endswith('.__value__')}

    @classmethod
    def from_file(cls, filepath:str):
        history = cls([])
        history.load(filepath)
        return history


    def xy(self, key:str | Any):
        if isinstance(key, str):
            # keys like `weights[0,1]` for indexing multidimensional arrays
            if key.count('[') == 1 and key.count(']') == 1 and key.endswith(']'):
                index: str | None = key[key.index('[') + 1 : key.index(']')]
                key = key[:key.index('[')]
            else: index = None

            if key.startswith(('params.', 'logs.', 'soft_violations.', 'hard_violations.', 'soft.', 'hard.')):
                if key.startswith('soft.'): key = key.replace('soft.', 'soft_violations.')
                if key.startswith('hard.'): key = key.replace('hard.', 'hard_violations.')
                keys_arr = self.data_keys[key]
                vals_arr = self.data_values[key]
            else:
                if key in self.data_keys:
                    keys_arr = self.data_keys[key]
                    vals_arr = self.data_values[key]
                elif key in self.aliases:
                    keys_arr = self.data_keys[self.aliases[key]]
                    vals_arr = self.data_values[self.aliases[key]]
                elif key == 'params':
                    keys_arr = list(range(len(self)))
                    values_arrs = [self.data_values[i] for i in self.data_keys if i.startswith('params.')]
                    values_arrs = [[i[int(j)].flatten() for j in keys_arr] for i in values_arrs]
                    values_arrs = [i for i in values_arrs if len(i) > 0 and np.issubdtype(i[0].dtype, np.number)]
                    vals_arr = np.concatenate(values_arrs, axis=1)
                elif f'params.{key}' in self.data_keys:
                    keys_arr = self.data_keys[f'params.{key}']
                    vals_arr = self.data_values[f'params.{key}']
                elif f'logs.{key}' in self.data_keys:
                    keys_arr = self.data_keys[f'logs.{key}']
                    vals_arr = self.data_values[f'logs.{key}']
                else: raise KeyError(f'Key {key} not found in history')

            if index is not None:
                index = index.lower().strip()
                if index == 'min': vals_arr = vals_arr.min(axis=tuple(range(1, vals_arr.ndim)))
                elif index == 'max': vals_arr = vals_arr.max(axis=tuple(range(1, vals_arr.ndim)))
                elif index in ('mean', 'avg', 'average'): vals_arr = vals_arr.mean(axis=tuple(range(1, vals_arr.ndim)))
                elif index == 'randmask':
                    shape = vals_arr.shape[1:]
                    mask = _randbinary_exact_proportion(0.5, size = shape)
                    vals_arr = np.mean(vals_arr * mask, axis=tuple(range(1, vals_arr.ndim)))
                elif index == 'randmask2':
                    shape = vals_arr.shape[1:]
                    if 2 in self._randmasks:
                        mask = 1 - self._randmasks[2]
                        del self._randmasks[2]
                    else: mask = self._randmasks[2] = _randbinary_exact_proportion(0.5, size = shape)
                    vals_arr = np.mean(vals_arr * mask, axis=tuple(range(1, vals_arr.ndim)))
                else:
                    slices = [int(i) for i in index.split(',')]
                    vals_arr = vals_arr[:, *slices]

        else:
            keys_arr = self.data_keys[key]
            vals_arr = self.data_values[key]

        return (keys_arr, vals_arr)

    def min(self, key: str | Any): return float(self[key].min())
    def max(self, key: str | Any): return float(self[key].max())
    def mean(self, key: str | Any): return float(self[key].mean())

    def __getitem__(self, key: str | Any):
        return self.xy(key)[1]

    def __contains__(self, key: str | Any):
        return key in self.data_keys

    def _reset(self):
        self._randmasks = {}

    def fill_nan(self, key) -> list[Any]:
        """Returns a dictionary of values per each evaluation the given key, but missing values are changed to nan."""
        x = list(range(0, len(self)))
        d = dict(zip(*self.xy(key)))
        return [(d[k] if k in d else nan) for k in x]

    def plot(self, key:str, ax=None):
        import matplotlib.pyplot as plt

        from .plt_tools import ax_plot_
        if ax is None: ax = plt.gca()
        ax_plot_(ax, *self.xy(key), xlabel=key, ylabel = 'eval')

    def plot_convergence(
        self,
        y: str = "value",
        best: Literal["min", "max"] = "min",
        x: Literal["eval", "step", "time"] | str = "eval",
        xlim = None,
        ylim = None,
        ax=None,
    ):
        self._reset()
        import matplotlib.pyplot as plt

        from optforge.plt_tools import ax_plot_, ax_scatter_
        if ax is None: ax = plt.gca()
        xvals, yvals = self.get_valid(x, y)# pylint:disable=W0632
        ax_scatter_(ax, xvals, yvals, xlabel=x, ylabel = y, s=3, alpha = 0.2, xlim=xlim, ylim=ylim)
        if best == 'min':  yacc = np.minimum.accumulate(yvals)
        elif best == 'max':yacc = np.maximum.accumulate(yvals)
        else: raise ValueError(f'best must be min or max, not {best}')
        ax_plot_(ax, xvals, yacc, color='red', alpha=0.5,xlim=xlim, ylim=ylim)
        if best == 'min': ax.fill_between(xvals, yacc, np.min(yacc), alpha = 0.25, color = 'red')
        elif best == 'max': ax.fill_between(xvals, yacc, np.max(yacc), alpha = 0.25, color = 'red')
        else: raise ValueError(f'best must be min or max, not {best}')

    def get_valid(self, *keys:str | Any | None) -> list[np.ndarray | Any]:
        """Finds all evaluations where all `keys` exist, returns lists of values for each key. Non string keys are returned as is."""
        arr = np.array([self.fill_nan(key) for key in keys if isinstance(key,str)]).T
        mask = np.ma.fix_invalid(arr).mask
        # if there are no nan values, mask will be a 0 ndim array
        if mask.ndim > 1: valid_arr = arr[~np.ma.fix_invalid(arr).mask.any(axis=1)].T
        else: valid_arr = arr.T
        valids = []
        i = 0
        for key in keys:
            if isinstance(key, str):
                valids.append(valid_arr[i])
                i += 1
            else: valids.append(key)
        return valids

    def get_valid_with_length(self, *keys:str | Any | None, length: int) -> list[np.ndarray | Any]:
        arrs = self.get_valid(*keys)
        valids = []
        i = 0
        for key in keys:
            if isinstance(key, str):
                if len(arrs[i] < length):
                    valids.append(np.concatenate((arrs[i], np.full(length - len(arrs[i]), np.nan))))
                else: valids.append(arrs[i][:length])
                i += 1
            else: valids.append(key)
        return valids

    def linechart(self, x:str, y:str, sort:Optional[str]=None, ax=None):
        self._reset()
        import matplotlib.pyplot as plt

        from .plt_tools import ax_plot_
        if ax is None: ax = plt.gca()
        xvals, yvals, sortvals = self.get_valid(x, y, sort) # pylint:disable=W0632
        if sort is not None:
            sortvals = [i for i,_ in sorted(enumerate(sortvals), key=lambda x:x[1])]
            xvals = [xvals[i] for i in sortvals]
            yvals = [yvals[i] for i in sortvals]
        ax_plot_(ax, xvals, yvals, xlabel=x, ylabel = y)

    def multi_linechart(self, x:Iterable[str], y:str, sort:Optional[str]=None, ax=None):
        self._reset()
        import matplotlib.pyplot as plt
        if ax is None: ax = plt.gca()
        for i in x: self.linechart(i, y, sort=sort, ax=ax)

    def scatter(
        self,
        x: str,
        y: str,
        c: Optional[str] = None,
        s: Optional[str | float] = 6,
        alpha: float = 0.5,
        norm=None,
        cmap=None,
        xlim=None,
        ylim=None,
        ax=None,
    ):
        self._reset()
        import matplotlib.pyplot as plt

        from .plt_tools import ax_scatter_
        if ax is None: ax = plt.gca()
        ax_scatter_( # pylint:disable = E1120
            ax,
            *self.get_valid(x, y, s, c),
            alpha=alpha,
            xlabel=x,
            ylabel=y,
            cmap=cmap,
            norm = norm,
            label=c,
            xlim=xlim,
            ylim=ylim,
        )

    @f2f_remove3_add1(scatter) # type:ignore
    def scatter_landscape( # pylint:disable = W1113
        self,
        *args,
        **kwargs,
    ):
        return self.scatter('params[randmask2]', 'params[randmask2]', *args, **kwargs)

    def scatter3d(
        self,
        x: str,
        y: str,
        z: str,
        c: Optional[str] = None,
        s: str | int = 20,
        cmap=None,
        ax: "Optional[Axes3D]" = None,
        fig=None,
    ):
        self._reset()
        import matplotlib.pyplot as plt

        from .plt_tools import ax_scatter3d_
        if ax is None:
            if fig is None: fig = plt.figure()
            ax = fig.add_subplot(projection='3d') # type:ignore
        ax_scatter3d_( # pylint:disable = E1120
            ax,  # type:ignore
            *self.get_valid(x, y, z, s, c), # type:ignore
            xlabel=x,
            ylabel=y,
            zlabel=z,
            cmap = cmap,
            title = c,
        )

    def contour(
        self,
        x: str,
        y: str,
        z: str,
        levels=12,
        num=300,
        filled = False,
        cmap=None,
        mode:Literal["linear", "nearest", "clough", 'kde']="linear",
        xlim=None,
        ylim=None,
        zlim=None,
        alpha=None,
        norm=None,
        kde_nbins = 100,
        grid = True,
        colorbar = True,
        ax = None,
    ):
        self._reset()
        import matplotlib.pyplot as plt

        from .plt_tools import ax_contour_
        if ax is None: ax = plt.gca()
        ax_contour_( # pylint:disable=E1120
            ax,
            *self.get_valid(x,y,z),
            levels = levels,
            num = num,
            filled = filled,
            cmap = cmap,
            mode = mode,
            xlim = xlim,
            ylim = ylim,
            zlim = zlim,
            alpha = alpha,
            norm = norm,
            kde_nbins = kde_nbins,
            grid = grid,
            colorbar=colorbar,
            xlabel=x,
            ylabel=y,
        )

    @f2f_remove3_add1(contour) # type:ignore
    def contour_landscape( # pylint:disable = W1113
        self,
        *args,
        **kwargs,
    ):
        return self.contour('params[randmask2]', 'params[randmask2]', *args, **kwargs)

    def pcolormesh(
        self,
        x: str,
        y: str,
        z: str,
        num=300,
        mode:Literal["linear", "nearest", "clough", 'kde']="linear",
        cmap='coolwarm',
        contour=True,
        contour_cmap="binary",
        contour_levels=12,
        contour_lw:float = 0.5,
        contour_alpha=0.2,
        scatter = False,
        scatter_s: Optional[str | float] = 4,
        scatter_c: Optional[str | Any] = None,
        scatter_cmap = None,
        scatter_alpha = 0.1,
        xlim=None,
        ylim=None,
        zlim=None,
        alpha=None,
        shading: Optional[Literal['flat', 'nearest', 'gouraud']]=None,
        norm = None,
        antialiased: bool = True,
        kde_nbins = 100,
        grid = True,
        colorbar = True,
        ax = None,
        figsize = None
    ):
        self._reset()
        import matplotlib.pyplot as plt

        from .plt_tools import ax_pcolormesh_, ax_scatter_
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        if scatter: xvals, yvals, zvals, cvals, svals = self.get_valid(x, y, z, scatter_c, scatter_s) # pylint:disable=W0632
        else: xvals, yvals, zvals = self.get_valid(x, y, z) # pylint:disable=W0632
        ax_pcolormesh_(
            ax = ax,
            x = xvals,
            y = yvals,
            z = zvals,
            num = num,
            cmap = cmap,
            contour = contour,
            contour_cmap = contour_cmap,
            contour_levels = contour_levels,
            contour_alpha = contour_alpha,
            contour_lw = contour_lw,
            mode = mode,
            xlim = xlim,
            ylim = ylim,
            zlim = zlim,
            alpha = alpha,
            shading = shading,
            norm = norm,
            antialiased = antialiased,
            kde_nbins=kde_nbins,
            grid = grid,
            colorbar = colorbar,
            xlabel=x,
            ylabel=y,
        )
        if scatter: ax_scatter_(
            ax = ax,
            x = xvals,
            y = yvals,
            c = cvals, # type:ignore
            s = svals, # type:ignore
            cmap=scatter_cmap,
            alpha = scatter_alpha,
            xlim = xlim,
            ylim = ylim,
        )

    @f2f_remove3_add1(pcolormesh) # type:ignore
    def pcolormesh_landscape( # pylint:disable = W1113
        self,
        *args,
        **kwargs,
    ):
        return self.pcolormesh('params[randmask2]', 'params[randmask2]', *args, **kwargs)

    def hist2d(
        self,
        x,
        y,
        bins: tuple[int,int] = (100,100),
        range: Optional[Sequence[Sequence[float]]] = None,
        density: bool = False,
        weights: Optional[Sequence | np.ndarray] = None,
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        cmap = None,
        norm = None,
        colorbar = True,
        ax: "Optional[Axes]" = None,
        figsize = None
    ):
        self._reset()
        import matplotlib.pyplot as plt
        from .plt_tools import ax_hist2d_
        if ax is None: fig, ax = plt.subplots(figsize=figsize)
        xvals, yvals = self.get_valid(x, y) # pylint:disable=W0632
        ax_hist2d_(
            ax = ax,
            x = xvals,
            y = yvals,
            bins = bins,
            range = range,
            density = density,
            weights = weights,
            cmin = cmin,
            cmax = cmax,
            cmap = cmap,
            norm = norm,
            colorbar=colorbar,
            xlabel = x,
            ylabel=y,
        )


    def surface(
        self,
        x: str,
        y: str,
        z: str,
        num=50,
        mode:Literal["linear", "nearest", "clough", 'kde']="linear",
        color = None,
        cmap='coolwarm',
        xlim=None,
        ylim=None,
        zlim=None,
        shade=True,
        norm = None,
        kde_nbins = 100,
        grid = True,
        colorbar = True,
        ax: "Optional[Axes3D]" = None,
        fig = None,
        figsize = None,
    ):
        self._reset()
        import matplotlib.pyplot as plt

        from .plt_tools import ax_surface_
        if ax is None:
            if fig is None: fig = plt.figure(figsize = figsize)
            ax = fig.add_subplot(projection='3d') # type:ignore
        ax_surface_( # pylint:disable=E1120
            ax, # type:ignore
            *self.get_valid(x,y,z),
            num = num,
            mode = mode,
            color = color,
            cmap = cmap,
            xlim = xlim,
            ylim = ylim,
            zlim = zlim,
            shade = shade,
            norm = norm,
            kde_nbins = kde_nbins,
            grid = grid,
            colorbar = colorbar,
            xlabel=x,
            ylabel=y,
            zlabel=z,
        )

