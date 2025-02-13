from collections.abc import Callable, Hashable, Sequence
from typing import Any, Literal, Optional, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import AutoLocator, AutoMinorLocator

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d.axes3d import Axes3D

def ax_ticks_(ax:Axes):
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

def ax_format_(
    ax: Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlim: Optional[tuple[float, float]] = None,
    ylim: Optional[tuple[float, float]] = None,
    grid: bool = True,
):
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax_ticks_(ax)
    if grid: ax_grid_(ax)
    if title is not None: ax.set_title(title)

def ax_grid_(ax:Axes, color='black', alphas = (0.08, 0.03), lws = (1,1)):
    ax.grid(which="major", color=color, alpha=alphas[0], lw = lws[0])
    ax.grid(which="minor", color=color, alpha=alphas[1], lw = lws[1])

def ax_plot_(ax:Axes, x, y, xlabel = None, ylabel = None, grid = True, color = None, alpha = 1., label=None, xlim=None, ylim=None, title = None, **kwargs):
    ax.plot(x, y, color = color, alpha = alpha, label=label, **kwargs)
    ax_format_(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, grid=grid)

def ax_scatter_(
    ax: Axes,
    x,
    y,
    s=None,
    c=None,
    xlabel=None,
    ylabel=None,
    alpha=None,
    grid=True,
    cmap=None,
    norm=None,
    label=None,
    xlim=None,
    ylim=None,
    title=None,
):
    ax.scatter(x = x, y = y, s = s, c = c, alpha = alpha, cmap = cmap, norm = norm, label=label)
    ax_format_(ax, title=title, xlabel=xlabel, ylabel=ylabel, xlim=xlim, ylim=ylim, grid=grid)

def ax_scatter3d_(ax:"Axes3D", x, y, z, s = 20, c = None, xlabel = None, ylabel = None, zlabel = None, alpha = None, grid = True, cmap = None, title=None):
    ax.scatter(xs = x, ys = y, zs = z, s = s, c = c, alpha = alpha, cmap = cmap, label=title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if ylabel is not None: ax.set_zlabel(zlabel)
    ax_ticks_(ax)
    if grid: ax_grid_(ax)

def _create_grid(x, y, z, mode:Literal["linear", "nearest", "clough", 'kde'], xlim, ylim, zlim, num, kde_nbins, ):
    xmin, xmax = np.min(x), np.max(x)
    if xlim is not None:
        if xlim[0] is not None: xmin = xlim[0]
        if xlim[1] is not None: xmax = xlim[1]
    ymin, ymax = np.min(y), np.max(y)
    if ylim is not None:
        if ylim[0] is not None: ymin = ylim[0]
        if ylim[1] is not None: ymax = ylim[1]
    if zlim is not None:
        z = np.clip(z, *zlim)

    if mode == 'kde':
        from scipy.stats import gaussian_kde
        x = np.array(x, copy=False); y = np.array(y, copy=False); z = np.array(z, copy=False)
        k = gaussian_kde(np.array([x, y]))

        X, Y = np.mgrid[x.min():x.max():kde_nbins*1j, y.min():y.max():kde_nbins*1j]
        Z = k(np.vstack([X.flatten(), Y.flatten()]))
        Z = Z.reshape(X.shape)
        return X, Y, Z
    else:
        from scipy.interpolate import (CloughTocher2DInterpolator,
                                    LinearNDInterpolator, NearestNDInterpolator)
        x_grid = np.linspace(xmin, xmax, num)
        y_grid = np.linspace(ymin, ymax, num)
        X, Y = np.meshgrid(x_grid,y_grid)
        INTERPOLATORS = {
            'linear': LinearNDInterpolator,
            'nearest': NearestNDInterpolator,
            'clough': CloughTocher2DInterpolator
        }
        if mode not in INTERPOLATORS: raise ValueError(f'Invalid mode {mode}')

        interpolator = INTERPOLATORS[mode]((x, y), z)
        Z = interpolator(X, Y)
        return X, Y, Z

def ax_colorbar_(ax:Axes, location = None, orientation = None, fraction=0.15, shrink = 1., aspect = 20.):
    plt.colorbar(ax.collections[0], location=location,orientation=orientation, fraction=fraction, shrink=shrink, aspect=aspect)

def ax_contour_(
    ax: Axes,
    x,
    y,
    z,
    levels=12,
    num=300,
    filled = False,
    cmap=None,
    mode: Literal["linear", "nearest", "clough", 'kde'] = "linear",
    xlim=None,
    ylim=None,
    zlim=None,
    alpha=None,
    norm=None,
    kde_nbins = 100,
    xlabel=None,
    ylabel=None,
    grid = True,
    colorbar = True,
    ):
    X, Y, Z = _create_grid(x, y, z, mode = mode, xlim = xlim, ylim = ylim, zlim = zlim, num = num, kde_nbins = kde_nbins, )
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax_ticks_(ax)
    if grid: ax_grid_(ax)
    if filled: ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha = alpha, norm = norm)
    else: ax.contour(X, Y, Z, levels=levels, cmap=cmap, alpha = alpha, norm = norm)
    if colorbar: ax_colorbar_(ax)

def ax_pcolormesh_(
    ax: Axes,
    x,
    y,
    z,
    num=300,
    cmap='coolwarm',
    contour=True,
    contour_cmap="binary",
    contour_levels=10,
    contour_lw:float = 0.5,
    contour_alpha=0.2,
    mode:Literal["linear", "nearest", "clough", 'kde']="linear",
    xlim=None,
    ylim=None,
    zlim=None,
    alpha=None,
    shading: Optional[Literal['flat', 'nearest', 'gouraud']]=None,
    norm = None,
    antialiased: bool = True,
    kde_nbins = 100,
    xlabel=None,
    ylabel=None,
    grid = False,
    colorbar = True,
    ):
    X, Y, Z = _create_grid(x, y, z, mode = mode, xlim = xlim, ylim = ylim, zlim = zlim, num = num, kde_nbins = kde_nbins, )

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax_ticks_(ax)
    if grid: ax_grid_(ax, 'gray', (0.1, 0.04), (0.5, 0.5))
    ax.pcolormesh(X, Y, Z, cmap=cmap, alpha = alpha, shading = shading, antialiased = antialiased, zorder=0, norm=norm)
    if contour: ax.contour(X, Y, Z, levels=contour_levels, cmap=contour_cmap, alpha=contour_alpha, norm=norm,linewidths=contour_lw)
    if colorbar: ax_colorbar_(ax)


def ax_surface_(
    ax: "Axes3D",
    x,
    y,
    z,
    num=50,
    mode:Literal["linear", "nearest", "clough", 'kde']="linear",
    color = None,
    cmap=None,
    xlim=None,
    ylim=None,
    zlim=None,
    shade=True,
    norm = None,
    kde_nbins = 100,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    antialiased: bool = True,
    grid = True,
    colorbar = True,
    ):
    X, Y, Z = _create_grid(x, y, z, mode = mode, xlim = xlim, ylim = ylim, zlim = zlim, num = num, kde_nbins=kde_nbins)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if zlabel is not None: ax.set_zlabel(zlabel)
    ax_ticks_(ax)
    if grid: ax_grid_(ax)
    ax.plot_surface(X, Y, Z,  color=color, cmap=cmap, shade = shade, norm=norm, antialiased=antialiased)
    if colorbar: ax_colorbar_(ax)

# 'outside upper left', 'outside upper center', 'outside upper right',
#    'outside center left', 'upper center right',
#    'outside lower left', 'outside lower center', 'outside lower right',

def ax_legend_(
    ax: Axes,
    loc=None,
    frameon=True,
    framealpha=0.5,
    edgecolor="gray",
    facecolor="white",
    markerscale=1.5,
    bbox_to_anchor = None,
    ncol = 1,
):
    ax.legend(
        fontsize=6,
        loc=loc,
        frameon=frameon,
        framealpha=framealpha,
        edgecolor=edgecolor,
        facecolor=facecolor,
        markerscale=markerscale,
        bbox_to_anchor = bbox_to_anchor,
        ncol = ncol,
    )


def ax_hist2d_(
    ax: Axes,
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
    xlabel=None,
    ylabel=None,
    colorbar = True,
):
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    ax_ticks_(ax)
    ax.hist2d(
        x = x,
        y = y,
        bins = bins,
        range = range,
        density = density,
        weights = weights,
        cmin = cmin,
        cmax = cmax,
        cmap = cmap,
        norm = norm,
    )
    if colorbar: ax_colorbar_(ax)