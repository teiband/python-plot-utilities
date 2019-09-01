# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import helper as hlp

#%%============================================================================
def plot_ranking(ranking, fig=None, ax=None, figsize='auto', dpi=100,
                 barh=True, top_n=None, score_ax_label=None, name_ax_label=None,
                 invert_name_ax=False, grid_on=True):
    '''
    Plot rankings as a bar plot (in descending order), such as::

                ^
                |
        dolphin |||||||||||||||||||||||||||||||
                |
        cat     |||||||||||||||||||||||||
                |
        rabbit  ||||||||||||||||
                |
        dog     |||||||||||||
                |
               -|------------------------------------>  Age of pet
                0  1  2  3  4  5  6  7  8  9  10  11

    Parameters
    ----------
    ranking : dict or pandas.Series
        The ranking information, for example:
            {'rabbit': 5, 'cat': 8, 'dog': 4, 'dolphin': 10}
        It does not need to be sorted externally.
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    barh : bool
        Whether or not to show the bars as horizontal (otherwise, vertical)
    top_n : int
        If ``None``, show all categories. ``top_n`` > 0 means showing the
        highest ``top_n`` categories. ``top_n`` < 0 means showing the lowest
        |``top_n``| categories.
    score_ax_label : str
        Label of the score axis (e.g., "Age of pet").
    name_ax_label : str
        Label of the "category name" axis (e.g., "Pet name").
    invert_name_ax : bool
        Whether to invert the "category name" axis. For example, if
        ``invert_name_ax`` is ``False``, then higher values are shown on the
        top if ``barh`` is ``True``.
    grid_on : bool
        Whether or not to show grids on the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    if not isinstance(ranking, (dict, pd.Series)):
        raise TypeError('`ranking` must be a Python dict or pandas Series.')

    if top_n is not None and not isinstance(top_n, (int, np.integer)):
        raise ValueError('`top_n` must be an integer of None.')

    if top_n == None:
        nr_classes = len(ranking)
        top_n = len(ranking)
    else:
        nr_classes = np.abs(top_n)

    if figsize == 'auto':
        if barh:
            figsize = (5, nr_classes * 0.26)  # 0.26 inch = height for each category
        else:
            figsize = (nr_classes * 0.26, 5)

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    if isinstance(ranking,dict):
        ranking = pd.Series(ranking)

    if barh:
        kind = 'barh'
        xlabel, ylabel = score_ax_label, name_ax_label
        ax = ranking.sort_values(ascending=(top_n >= 0)).iloc[-np.abs(top_n):]\
                    .plot(kind=kind, ax=ax)
    else:
        kind = 'bar'
        xlabel, ylabel = name_ax_label, score_ax_label
        ax = ranking.sort_values(ascending=(top_n < 0))\
                    .iloc[:np.abs(top_n) if top_n != 0 else None]\
                    .plot(kind=kind, ax=ax)

    if invert_name_ax:
        if barh is True:
            ax.invert_yaxis()
        else:
            ax.invert_xaxis()
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid_on:
        ax.grid(ls=':')
        ax.set_axisbelow(True)

    return fig, ax

#%%============================================================================
def plot_with_error_bounds(x, y, upper_bound, lower_bound,
                           fig=None, ax=None, figsize=None, dpi=100,
                           line_color=[0.4]*3, shade_color=[0.7]*3,
                           shade_alpha=0.5, linewidth=2.0, legend_loc='best',
                           line_label='Data', shade_label='$\mathregular{\pm}$STD',
                           logx=False, logy=False, grid_on=True):
    '''
    Plot a graph with one line and its upper and lower bounds, with areas between
    bounds shaded. The effect is similar to this illustration below::

      y ^            ...                         _____________________
        |         ...   ..........              |                     |
        |         .   ______     .              |  ---  Mean value    |
        |      ...   /      \    ..             |  ...  Error bounds  |
        |   ...  ___/        \    ...           |_____________________|
        |  .    /    ...      \    ........
        | .  __/   ...  ....   \________  .
        |  /    ....       ...          \  .
        | /  ....            .....       \_
        | ...                    ..........
       -|--------------------------------------->  x


    Parameters
    ----------
    x : list, numpy.ndarray, or pandas.Series
        X data points to be plotted as a line.
    y : list, numpy.ndarray, or pandas.Series
        Y data points to be plotted as a line.
    fig : matplotlib.figure.Figure or ``None``
        Figure object. If None, a new figure will be created.
    ax : matplotlib.axes._subplots.AxesSubplot or ``None``
        Axes object. If None, a new axes will be created.
    figsize: (float, float)
        Figure size in inches, as a tuple of two numbers. The figure
        size of ``fig`` (if not ``None``) will override this parameter.
    dpi : float
        Figure resolution. The dpi of ``fig`` (if not ``None``) will override
        this parameter.
    upper_bound : list, numpy.ndarray, or pandas.Series
        Upper bound of the Y values.
    lower_bound : list, numpy.ndarray, or pandas.Series
        Lower bound of the Y values.
    line_color : str, list, or tuple
        Color of the line.
    shade_color : str, list, or tuple
        Color of the underlying shades.
    shade_alpha : float
        Opacity of the shades.
    linewidth : float
        Width of the line.
    legend_loc : int, str
        Location of the legend, to be passed directly to ``plt.legend()``.
    line_label : str
        Label of the line, to be used in the legend.
    shade_label : str
        Label of the shades, to be used in the legend.
    logx : bool
        Whether or not to show the X axis in log scale.
    logy : bool
        Whether or not to show the Y axis in log scale.
    grid_on : bool
        Whether or not to show grids on the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object being created or being passed into this function.
    ax : matplotlib.axes._subplots.AxesSubplot
        The axes object being created or being passed into this function.
    '''
    if not isinstance(x, hlp._array_like) or not isinstance(y, hlp._array_like):
        raise TypeError('`x` and `y` must be arrays.')

    if len(x) != len(y):
        raise hlp.LengthError('`x` and `y` must have the same length.')

    fig, ax = hlp._process_fig_ax_objects(fig, ax, figsize, dpi)

    hl1 = ax.fill_between(x, lower_bound, upper_bound,
                           color=shade_color, facecolor=shade_color,
                           linewidth=0.01, alpha=shade_alpha, interpolate=True,
                           label=shade_label)
    hl2, = ax.plot(x, y, color=line_color, linewidth=linewidth, label=line_label)
    if logx: ax.set_xscale('log')
    if logy: ax.set_yscale('log')

    if grid_on:
        ax.grid(ls=':',lw=0.5)
        ax.set_axisbelow(True)

    plt.legend(handles=[hl2,hl1],loc=legend_loc)

    return fig, ax

