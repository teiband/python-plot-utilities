# -*- coding: utf-8 -*-
"""
This is a Python module that contains some useful plotting utilities.

For user guide, check: https://github.com/jsh9/python-plot-utilities

Created on Fri Apr 28 15:37:26 2017

Copyright (c) 2017, Jian Shi
License: GPL v3
"""

import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#%%----------------------------------------------------------------------------
__version__ = '0.3.0'

#%%----------------------------------------------------------------------------
if sys.version_info.major == 3:  # Python 3
    unicode = str  # define 'unicode' as type name

#%%============================================================================
def process_fig_ax_objects(fig, ax, figsize=None, dpi=None, ax_proj=None):
    '''
    Process figure and axes objects. If fig and ax are None, create new figure
    and new axes according to figsize, dpi, and ax_proj (axes projection). If
    fig and ax are not None, use the passed-in fig and/or ax.

    Returns fig and ax back to its caller.
    '''

    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=figsize,dpi=dpi)
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes(projection=ax_proj)  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    return fig, ax

#%%============================================================================
def category_means(x, y, fig=None, ax=None, figsize=(3,3),
                   dpi=100, title=None, xlabel=None, ylabel=None, rot=0,
                   show_stats=True, **violinplot_kwargs):
    '''
    Summarize the mean values of entries of y corresponding to each distinct
    category in x, and show a violin plot to visualize it. The violin plot will
    show the distribution of y values corresponding to each category in x.

    Also, a one-way ANOVA test (H0: different categories in x yield same
    average y values) is performed, and F statistics and p-value are returned.

    Parameters
    ----------
    x : <array_like>
        An vector of categorical values.
    y : <array_like>
        The target variable whose values correspond to the values in x. Must
        have the same length as x. It is natural that y contains continuous
        values, but if y contains categorical values (expressed as integers,
        not strings), this function should also work.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    title : <str>
        The title of the violin plot, usually the name of vector x.
    xlabel : <str>
        The label for the x axis (i.e., categories) of the violin plot. If None
        and x is a pandas Series, use x's 'name' attribute as xlabel.
    ylabel : <str>
        The label for the y axis (i.e., average y values) of the violin plot.
        If None and y is a pandas Series, use y's 'name' attribute as ylabel.
    rot : <float>
        The rotation (in degrees) of the x axis labels.
    show_stats : <bool>
        Whether or not to show the statistical test results (F statistics
        and p-value) on the figure.
    **violinplot_kwargs :
        Keyword arguments to be passed to plt.violinplot().
        (https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.violinplot.html)
        Note that this subroutine overrides the default behavior of violinplot:
        showmeans is overriden to True and showextrema to False.

    Return
    ------
    fig, ax :
        Figure and axes objects
    mean_values : <dict>
        A dictionary whose keys are the categories in x, and their corresponding
        values are the mean values in y.
    F_test_result : <tuple>
        A tuple in the order of (F_stat, p_value), where F_stat is the computed
        F-value of the one-way ANOVA test, and p_value is the associated
        p-value from the F-distribution.
    '''

    if not isinstance(x, (pd.Series, np.ndarray, list)):
        raise TypeError('The input "x" must be pd.Series, np.array, or list.')
    if not isinstance(y, (pd.Series, np.ndarray, list)):
        raise TypeError('The input "y" must be pd.Series, np.array, or list.')
    if len(x) != len(y):
        raise ValueError('Lengths of x and y must be the same.')
    if isinstance(x, np.ndarray) and len(x.shape) > 1:
        raise TypeError('"x" must be a 1D numpy array. Please flatten it.')
    if isinstance(y, np.ndarray) and len(y.shape) > 1:
        raise TypeError('"y" must be a 1D numpy array. Please flatten it.')

    if not xlabel and isinstance(x, pd.Series): xlabel = x.name
    if not ylabel and isinstance(y, pd.Series): ylabel = y.name

    if isinstance(x, (list, pd.Series)): x = np.array(x)
    if isinstance(y, (list, pd.Series)): y = np.array(y)

    if pd.isnull(x).any():
        raise TypeError('"x" should not contain any None or NaN values.')

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    x_classes = list(np.unique(x))
    y_values = []  # each element in y_values represent the values of a category
    mean_values = {}  # each entry in the dict corresponds to a category
    for cat in x_classes:
        cat_index = np.where(x == cat)[0]
        y_values.append(list(y[cat_index]))
        mean_values[cat] = np.nanmean(y[cat_index])

    F_stat, p_value = stats.f_oneway(*y_values)  # pass every group into f_oneway()

    if 'showextrema' not in violinplot_kwargs:
        violinplot_kwargs['showextrema'] = False  # override default behavior of violinplot
    if 'showmeans' not in violinplot_kwargs:
        violinplot_kwargs['showmeans'] = True

    ax.violinplot(y_values, **violinplot_kwargs)

    ax.grid(ls=':')
    ax.set_axisbelow(True)

    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.set_xticks(range(1,len(x_classes)+1))

    ha = 'center' if 0 <= rot < 30 else 'right'
    ax.set_xticklabels([str(_) for _ in x_classes], rotation=rot, ha=ha)

    if show_stats:
        ax.annotate('F=%.2f, p_val=%.2g' % (F_stat, p_value),
                    xy=(0.05, 0.92), xycoords='axes fraction')

    if title: ax.set_title(title)

    return fig, ax, mean_values, (F_stat, p_value)

#%%============================================================================
def positive_rate(x, y, fig=None, ax=None, figsize='auto', dpi=100, barh=True,
                  top_n=-1, xlabel='Positive rate', ylabel='Categories',
                  show_stats=True):
    '''
    Calculate the proportions of the different categories in vector x that fall
    into class "1" (or "True") in vector y, and optionally show a figure.

    Also, a Pearson's chi-squared test is performed to test the independence
    between x and y. The chi-squared statistics, p-value, and degree-of-freedom
    are returned.

    Parameters
    ----------
    x : <array_like>
        An vector of categorical values.
    y : <array_like>
        The target variable of two classes. Each value in y correspond to a
        value in x (at the same index). Must have the same length as x.
        The second unique value in y will be considered as the positive class
        (for example, "True" in [True, False, True], or "3" in [1,1,3,3,1]).
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars, or 'auto'
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If 'auto', the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    barh : <bool>
        Whether or not to show the bars as horizontal (otherwise, vertical).
    top_n : <int>
        Only shows top_n categories (ranked by their positive rate) in the
        figure. Useful when there are too many categories.
    show_stats : <bool>
        Whether or not to show the statistical test results (chi2 statistics
        and p-value) on the figure.

    Returns
    -------
    fig, ax :
        Figure and axes objects
    pos_rate : <pd.Series>
        The positive rate of each categories in x.
    chi2_results : <tuple>
        A tuple in the order of (chi2, p_value, degree_of_freedom)
    '''
    import collections

    if not isinstance(x, (pd.Series, np.ndarray, list)):
        raise TypeError('The input "x" must be pd.Series, np.array, or list.')
    if not isinstance(y, (pd.Series, np.ndarray, list)):
        raise TypeError('The input "y" must be pd.Series, np.array, or list.')
    if len(x) != len(y):
        raise ValueError('Lengths of x and y must be the same.')
    if len(np.unique(y)) != 2:
        raise ValueError('y must contain only 2 distinct values.')
    if isinstance(x, np.ndarray) and len(x.shape) > 1:
        raise TypeError('"x" must be a 1D numpy array. Please flatten it.')
    if isinstance(y, np.ndarray) and len(y.shape) > 1:
        raise TypeError('"y" must be a 1D numpy array. Please flatten it.')

    if isinstance(x, (list, pd.Series)):
        x = np.array(x)
    if isinstance(y, (list, pd.Series)):
        y = np.array(y)

    if pd.isnull(x).any():
        raise TypeError('"x" should not contain any None or NaN values.')
    if pd.isnull(y).any():
        raise TypeError('"y" should not contain any None or NaN values.')

    nr_classes = len(np.unique(x))
    y_classes = list(np.unique(y))
    y_pos_index = np.where(y == y_classes[1])[0]

    count_all_classes = collections.Counter(x)
    count_pos_class = collections.Counter(x[y_pos_index])

    pos_rate = pd.Series(count_pos_class)/pd.Series(count_all_classes)
    pos_rate = pos_rate.fillna(0.0)  # keys not in count_pos_class show up as NaN

    observed = pd.crosstab(y, x)
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)

    if figsize == 'auto':
        if barh:
            figsize = (5, nr_classes * 0.26)  # 0.26 inch = height for each category
        else:
            figsize = (nr_classes * 0.26, 5)

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)
    fig, ax = plot_ranking(pos_rate, fig=fig, ax=ax, top_n=top_n, barh=barh,
                           score_ax_label=xlabel, name_ax_label=ylabel)

    if show_stats:
        ax.annotate('chi^2=%.2f, p_val=%.2g' % (chi2, p_val), ha='right',
                    xy=(0.99, 1.05), xycoords='axes fraction', va='bottom')

    return fig, ax, pos_rate, (chi2, p_val, dof)

#%%============================================================================
def contingency_table(array_horizontal, array_vertical,
                      fig=None, ax=None, figsize='auto', dpi=100,
                      color_map='auto', relative_color=True, show_stats=True):

    '''
    Calculate and visualize the contingency table from two categorical arrays.
    Also perform a Pearson's chi-squared test to evaluate whether the two arrays
    are independent.

    Parameters
    ----------
    array_horizontal : <array_like>
        Array to show as the horizontal margin in the contigency table (i.e.,
        its categories are the column headers)
    array_vertical : <array_like>
        Array to show as the vertical margin in the contigency table (i.e.,
        its categories are the row names)
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars, or 'auto'
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If 'auto', the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    color_map : <str> or <matplotlib.colors.Colormap>
        The color scheme specifications. Valid names are listed in
        https://matplotlib.org/users/colormaps.html.
        If relative_color is True, use diverging color maps (e.g., PiYG, PRGn,
        BrBG, PuOr, RdGy, RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr,
        seismic). Otherwise, use sequential color maps (e.g., viridis, jet).
    relative_color : <bool>
        Whether to show the contingency table as the original "observed
        frequency", or the relative difference (i.e., (obs. - exp.)/exp. ).
    show_stats : <bool>
        Whether or not to show the statistical test results (chi2 statistics
        and p-value) on the figure.

    Returns
    -------
    fig, ax :
        Figure and axes objects
    chi2_results : <tuple>
        A tuple in the order of (chi2, p_value, degree_of_freedom)
    '''

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    x = array_horizontal
    y = array_vertical

    if not isinstance(x, (pd.Series, np.ndarray, list)):
        raise TypeError('The input "x" must be pd.Series, np.array, or list.')
    if not isinstance(y, (pd.Series, np.ndarray, list)):
        raise TypeError('The input "y" must be pd.Series, np.array, or list.')
    if len(x) != len(y):
        raise ValueError('Lengths of x and y must be the same.')
    if isinstance(x, np.ndarray) and len(x.shape) > 1:
        raise TypeError('"x" must be a 1D numpy array. Please flatten it.')
    if isinstance(y, np.ndarray) and len(y.shape) > 1:
        raise TypeError('"y" must be a 1D numpy array. Please flatten it.')

    if isinstance(x, (list, np.ndarray)):
        x = pd.Series(x)
    if isinstance(y, (list, np.ndarray)):
        y = pd.Series(y)

    observed = pd.crosstab(y, x)
    chi2, p_val, dof, expected = stats.chi2_contingency(observed)
    expected = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    diff = (observed - expected) / expected

    if figsize == 'auto':
        figsize = observed.shape

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if relative_color:
        table = diff
    else:
        table = observed

    if color_map == 'auto':
        color_map = 'RdBu_r' if relative_color else 'viridis'

    im = ax.matshow(table, cmap=color_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)  # 'cb' is a Colorbar instance
    if relative_color:
        cb.set_label('(O-E)/E')
    else:
        cb.set_label('observed freq.')

    ax.set_xticks(range(table.shape[1]))
    ax.set_yticks(range(table.shape[0]))
    ax.set_xticklabels(table.columns)
    ax.set_yticklabels(table.index)

    if show_stats:
        ax.annotate('chi^2=%.2f, p_val=%.2g' % (chi2, p_val), ha='center',
                    xy=(0.5, -0.09), xycoords='axes fraction', va='top')

    tables = (observed, expected, diff)
    chi2_results = (chi2, p_val, dof)

    return fig, ax, tables, chi2_results

#%%============================================================================
def plot_ranking(ranking, fig=None, ax=None, figsize='auto', dpi=100,
                 barh=True, top_n=-1, score_ax_label=None, name_ax_label=None,
                 grid_on=True):
    '''
    Plots rankings as a bar plot (in descending order), such as:

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
    ranking : <dict> or <pd.Series>
        The ranking information, for example:
            {'rabbit': 5, 'cat': 8, 'dog': 4, 'dolphin': 10}
        It does not need to be sorted externally.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars, or 'auto'
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If 'auto', the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    barh : <bool>
        Whether or not to show the bars as horizontal (otherwise, vertical).
    top_n : <int>
        Only shows top_n categories (ranked by their positive rate) in the
        figure. Useful when there are too many categories.
    score_ax_label : <str>
        Label of the score axis (e.g., 'Age of pet').
    name_ax_label : <str>
        Label of the category name axis.
    grid_on : <bool>
        Whether or not to show grids.

    Returns
    -------
    fig, ax :
        Figure and axes objects
    '''

    if not isinstance(ranking, (dict, pd.Series)):
        raise TypeError('"ranking" must be a Python dict or pandas Series.')

    if top_n == -1:
        nr_classes = len(ranking)
    elif (top_n >= 1) and (isinstance(top_n,(int,np.integer))):
        nr_classes = top_n
    else:
        raise ValueError('top_n must be -1 or positive integers.')

    if figsize == 'auto':
        if barh:
            figsize = (5, nr_classes * 0.26)  # 0.26 inch = height for each category
        else:
            figsize = (nr_classes * 0.26, 5)

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if isinstance(ranking,dict):
        ranking = pd.Series(ranking)

    if barh == True:
        kind = 'barh'
        xlabel, ylabel = score_ax_label, name_ax_label
    else:
        kind = 'bar'
        xlabel, ylabel = name_ax_label, score_ax_label

    if top_n == -1:  # plot all categories
        ax = ranking.sort_values(ascending=barh).plot(kind=kind,ax=ax)
    elif (top_n >= 1) and (isinstance(top_n,(int,np.integer))):
        ax = ranking.sort_values(ascending=barh).iloc[-top_n:].plot(kind=kind,ax=ax)
    else:
        raise ValueError('top_n must be -1 or positive integers.')
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if grid_on == True:
        ax.grid(ls=':')
        ax.set_axisbelow(True)

    return fig, ax

#%%============================================================================
def missing_value_counts(X, fig=None, ax=None, figsize=(12,3), dpi=100, rot=90):
    '''
    Visualize the number of missing values in each column of X.

    Parameters
    ----------
    X : <pd.DataFrame> or <pd.Series>
        Input data set whose every row is an observation and every column is
        a variable.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    rot : <float>
        Rotation (in degrees) of the x axis labels

    Returns
    -------
    fig, ax :
        Figure and axes objects
    null_counts : <pd.Series>
        A pandas Series whose every element is the number of missing values
        corresponding to each column of X.
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if not isinstance(X, (pd.DataFrame, pd.Series)):
        raise TypeError('X should be pandas DataFrame or Series.')

    if isinstance(X, pd.Series):
        X = pd.DataFrame(X)

    ncol = X.shape[1]
    null_counts = X.isnull().sum()  # a pd Series containing number of non-null numbers

    ax.bar(range(ncol), null_counts)
    ax.set_xticks(range(ncol))

    ha = 'center' if 0 <= rot < 30 else 'right'
    ax.set_xticklabels(null_counts.index, rotation=rot, ha=ha)
    plt.ylabel('Number of missing values')
    plt.grid(ls=':')
    ax.set_axisbelow(True)

    alpha = null_counts.max()*0.02  # vertical offset for the texts

    for j, col in enumerate(null_counts.index):
        if null_counts[col] != 0:  # show count of missing values on top of bars
            plt.text(j, null_counts[col] + alpha, str(null_counts[col]),
                     ha='center', va='bottom', rotation=90)

    return fig, ax, null_counts

#%%============================================================================
def piechart(target_array, class_names=None, fig=None, ax=None,
             figsize=(3,3), dpi=100, colors=None, display='percent',
             fontsize=None, **piechart_kwargs):
    '''
    Plot a pie chart demonstrating proportions of different categories within
    an array.

    Parameters
    ----------
    target_array : <array_like>
        An array containing categorical values (could have more than two
        categories). Target value can be numeric or texts.
    class_names : sequence of str
        Names of different classes. The order should correspond to that in the
        target_array. For example, if target_array has 0 and 1 then class_names
        should be ['0', '1']; and if target_array has "pos" and "neg", then
        class_names should be ['neg','pos'] (i.e., alphabetical).
        If None, values of the categories will be used as names.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    colors : <list> or None
        A list of colors (can be RGB values, hex strings, or color names) to be
        used for each class. The length can be longer or shorter than the number
        of classes. If longer, only the first few colors are used; if shorter,
        colors are wrapped around.
        If None, automatically use the Pastel2 color map (8 colors total).
    display : ['percent', 'count', 'both', None]
        An option of what to show on top of each pie slices: percentage of each
        class, or count of each class, or both percentage and count, or nothing.
    fontsize : scalar or tuple/list of two scalars
        Font size. If scalar, both the class names and the percentages are set
        to the specified size. If tuple of two scalars, the first value sets
        the font size of class names, and the last value sets the font size
        of the percentages.
    **piechart_kwargs :
        Keyword arguments to be passed to matplotlib.pyplot.pie function,
        except for "colors", "labels"and  "autopct" because this subroutine
        re-defines these three arguments.
        (See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html)

    Returns
    -------
    fig, ax :
        Figure and axes objects
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if not isinstance(target_array,(np.ndarray,pd.DataFrame,pd.Series,list)):
        raise TypeError('Unrecognized data type for target_array.')
    y = target_array  # short hand

    if any(pd.isnull(np.array(y))):
        print('*****  WARNING: target_array contains NaN''s.  *****')

    vals = np.unique(y)  # vals is sorted by np.unique()
    x = []  # x stores the counts of different categories in y
    for val in vals:
        if pd.isnull(val):
            x.append(np.sum(pd.isnull(y)))  # count number of NaN's in y
        else:
            x.append(np.sum(y == val))

    if not colors:  # set default color cycle to 'Pastel2'
        colors_4 = mpl.cm.Pastel2(range(8))  # R,G,B,A values ("8" means Pastel2 has maximum 8 colors)
        colors = [list(_)[:3] for _ in colors_4]  # remove the fourth value

    if not class_names:
        class_names = [str(val) for val in vals]

    if display == 'percent':
        autopct = '%1.1f%%'
    elif display == 'count':
        total = np.sum(x)  # https://stackoverflow.com/a/14171272/8892243
        autopct = lambda p: '{:.0f}'.format(p * total / 100.0)
    elif display == 'both':
        def make_autopct(values):  # https://stackoverflow.com/a/6170354/8892243
            def my_autopct(pct):
                total = sum(values)
                val = int(round(pct*total/100.0))
                return '{p:.1f}%  ({v:d})'.format(p=pct,v=val)
            return my_autopct
        autopct = make_autopct(x)
    elif display == None:
        autopct = ''
    else:
        raise ValueError('Invalid value of "display". Can only be ["percent","count","both",None].')

    _,texts,autotexts = ax.pie(x,labels=class_names,colors=colors,
                               autopct=autopct,**piechart_kwargs)
    if isinstance(fontsize,(list,tuple)):
        for t_ in texts: t_.set_fontsize(fontsize[0])
        for t_ in autotexts: t_.set_fontsize(fontsize[1])
    elif fontsize:
        for t_ in texts: t_.set_fontsize(fontsize)
        for t_ in autotexts: t_.set_fontsize(fontsize)

    ax.axis('equal')

    return fig, ax

#%%############################################################################
def histogram3d(X, bins=10, fig=None, ax=None, figsize=(8,4), dpi=100,
                elev=30, azim=5, alpha=0.6, data_labels=None,
                plot_legend=True, plot_xlabel=False, color=None,
                dx_factor=0.6, dy_factor=0.8,
                ylabel='Data', zlabel='Counts',
                **legend_kwargs):
    '''
    Plot 3D histograms. 3D histograms are best used to compare the distribution
    of more than one set of data.

    Parameters
    ----------
    X :
        Input data. X can be:
           (1) a 2D numpy array, where each row is one data set;
           (2) a 1D numpy array, containing only one set of data;
           (3) a list of lists, e.g., [[1,2,3],[2,3,4,5],[2,4]], where each
               element corresponds to a data set (can have different lengths);
           (4) a list of 1D numpy arrays.
               [Note: Robustness is not guaranteed for X being a list of
                      2D numpy arrays.]
    bins : <int> or <array_like>
        Bin specifications. Can be:
           (1) An integer, which indicates number of bins;
           (2) An array or list, which specifies bin edges.
               [Note: If an integer is used, the widths of bars across data
                      sets may be different. Thus array/list is recommended.]
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    elev, azim : scalars
        Elevation and azimuth (3D projection view points)
    alpha : scalar
        Opacity of bars
    data_labels : list of <str>
        Names of different datasets, e.g., ['Simulation', 'Measurement'].
        If not provided, generic names ['Dataset #1', 'Dataset #2', ...]
        are used. The data_labels are only shown when either plot_legend or
        plot_xlabel is True.
    plot_legend : <bool>
        Whether to show legends or not
    plot_xlabel : <str>
        Whether to show data_labels of each data set on their respective x
        axis position or not
    color : list of lists, or tuple of tuples
        Colors of each distributions. Needs to be at least the same length as
        the number of data series in X. Can be RGB colors, HEX colors, or valid
        color names in Python. If None, get_colors('tab10',N=N) will be queried.
    dx_factor, dy_factor : scalars
        Width factor 3D bars in x and y directions. For example, if dy_factor
        is 0.9, there will be a small gap between bars in y direction.
    ylabel, zlabel : <str>
        Labels of y and z axes

    Returns
    -------
    fig, ax :
        Figure and axes objects

    Notes on x and y directions
    ---------------------------
        x direction: across data sets (i.e., if we have three datasets, the
                     bars will occupy three different x values)
        y direction: within dataset

    '''

    from mpl_toolkits.mplot3d import Axes3D

    # --------  Data type checking for X  ------------
    if isinstance(X,np.ndarray):
        if X.ndim <= 1:  # X is a 1D numpy array
            N = 1
            X = [list(X)]  # e.g., np.array([1,2,3]) --> [[1,2,3]], so that X[0] = [1,2,3]
        elif X.ndim == 2:  # X is a 2D numpy array
            N = X.shape[0]  # number of separate distribution to be compared
            X = list(X)  # then turn X into a list of numpy arrays (no longer a 2D numpy array)
        else:  # 3D numpy array or above
            print('*****  If X is a numpy array, it should be a 1D or 2D array.  *****')
            sys.exit()
    elif len(list(X)) > 1:  # adding list() to X to make sure len() does not throw an error
        N = len(X)  # number of separate distribution to be compared
    else:  # X is a scalar
        print('*****  X should either be a list or a 2D numpy array.  *****')
        sys.exit()

    # -----------  NaN checking for X  ----------------
    for j in range(N):
        if not all(np.isfinite(X[j])):
            raise ValueError('X[%d] contains non-finite values (not accepted by histogram3d()).' % j)

    if data_labels is None:
        data_labels = [[None]] * N
        for j in range(N):
            data_labels[j] = 'Dataset #%d' % (j+1)  # use generic data set names

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi, '3d')

    ax.view_init(elev,azim)  # set view elevation and angle

    proxy = [[None]] * N  # create a 'proxy' to help generating legends
    if not color:
        c_ = get_colors('tab10',N)  # get a list of colors
    else:
        valid_color_flag, msg = check_color_types(color, N)
        if not valid_color_flag:
            raise TypeError(msg)
        c_ = color
    xpos_list = [[None]] * N  # pre-allocation
    for j in range(N):  # loop through each dataset
        if isinstance(bins,(list,np.ndarray)) and len(bins) > 1:  # if 'bins' is a list and length > 1
            bar_width = np.min(np.array(bins[1:])-np.array(bins[:-1]))  # pick the mininum bin width as bar_width
        elif isinstance(bins,(int,np.integer)):  # if integer (i.e., number of bins)
            bar_width = (np.max(X[j])-np.min(X[j]))/float(bins)  # use the most narrow bin width as bar_width
        else:  # for other type of "bins", try to convert it into a list
            bins = list(bins)
            bar_width = np.min(np.array(bins[1:])-np.array(bins[:-1]))  # pick the mininum bin width as bar_width

        dz, ypos_ = np.histogram(X[j],bins)  # calculate counts and bin edges
        ypos = np.mean(np.array([ypos_[:-1],ypos_[1:]]),axis=0)  # mid-point of all bins
        xpos = np.ones_like(ypos) * (j-0.5)  # location of each data set
        zpos = np.zeros_like(xpos)  # should be all 0's
        dx = bar_width * dx_factor  # width of bars in x direction (across data sets)
        dy = bar_width * dy_factor  # width of bars in y direction (within data set)
        if float(mpl.__version__.split('.')[0]) >= 2.0:
            bar3d_kwargs = {'alpha':alpha}  # 'lw' argument clashes with alpha in 2.0+ versions
        else:
            bar3d_kwargs = {'alpha':alpha, 'lw':0.5}
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=c_[j], **bar3d_kwargs)
        proxy[j] = plt.Rectangle((0, 0), 1, 1, fc=c_[j])  # generate proxy for plotting legends
        xpos_list[j] = xpos[0] + dx/2.0  # '+dx/2.0' makes x ticks pass through center of bars

    if plot_legend is True:
        default_kwargs = {'loc':9, 'fancybox':True, 'framealpha':0.5,
                          'ncol':N, 'fontsize':10}
        if legend_kwargs == {}:  # set default legend keyword arguments
            legend_kwargs.update(default_kwargs)  # update with default_kwargs
        else:  # if user provides (some of the) keyword arguments
            default_kwargs.update(legend_kwargs)  # use provided args to update default args
            legend_kwargs = default_kwargs  # then replace legend_kwargs with default_kwargs
        ax.legend(proxy,data_labels,**legend_kwargs)

    if plot_xlabel is True:
        ax.set_xticks(xpos_list)  # show x ticks
        ax.set_xticklabels(data_labels)  # use data_labels to denote X ticks
    else:
        ax.set_xticks([])  # do not show X ticks and X tick labels

    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.invert_xaxis()  # make X[0] appear in front, and X[-1] appear at back

    plt.tight_layout(pad=0.3)

    return fig, ax

#%%############################################################################
def check_color_types(color, n=None):
    '''
    Helper function that checks whether a Python object "color" is indeed a
    valid list (or tuple) of length n that defines n colors.

    Returns True (valid) or False (otherwise), and an error message (empty
    message if True).
    '''
    if not isinstance(color,(list,tuple)):
        is_valid = False
        err_msg = '"color" must be a list of lists (or tuple of tuples).'
    elif not all([isinstance(c_,(list,tuple)) for c_ in color]) and \
         not all([isinstance(c_,(str,unicode)) for c_ in color]):
            is_valid = False
            err_msg = '"color" must be a list of lists (or tuple of tuples).'
    else:
        if n and len(color) < n:
            is_valid = False
            err_msg = 'Length of "color" must be at least the same length as "n".'
        else:
            is_valid = True
            err_msg = ''

    return is_valid, err_msg

#%%############################################################################
def get_colors(color_scheme='tab10',N=None):
    '''
    Returns a list of N distinguisable colors. When N is larger than the color
    scheme capacity, the color cycle is wrapped around.

    What does each color_scheme look like?
        https://matplotlib.org/mpl_examples/color/colormaps_reference_04.png
        https://matplotlib.org/users/dflt_style_changes.html#colors-color-cycles-and-color-maps
        https://github.com/vega/vega/wiki/Scales#scale-range-literals
        https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html

    Parameters
    ----------
    color_scheme : <str> or {8.3, 8.4}
        Color scheme specifier. Valid specifiers are:
        (1) Matplotlib qualitative color map names:
            'Pastel1'
            'Pastel2'
            'Paired'
            'Accent'
            'Dark2'
            'Set1'
            'Set2'
            'Set3'
            'tab10'
            'tab20'
            'tab20b'
            'tab20c'
            ![](https://matplotlib.org/mpl_examples/color/colormaps_reference_04.png)
        (2) 8.3 and 8.4 (floats): old and new MATLAB color scheme
            Old:
            ![](https://www.mathworks.com/help/matlab/graphics_transition/transition_colororder_old.png)
            New:
            ![](https://www.mathworks.com/help/matlab/graphics_transition/transition_colororder.png)
        (3) 'rgbcmyk': old default Matplotlib color palette (v1.5 and earlier)
        (4) 'bw' (or 'bw3'), 'bw4', and 'bw5'
            Black-and-white (grayscale colors in 3, 4, and 5 levels)
    N : <int> or None
        Number of qualitative colors desired. If None, returns all the colors
        in the specified color scheme.

    Returns
    -------
    A list of colors.
    '''

    nr_c = {'Pastel1': 9,  # number of qualitative colors in each color map
            'Pastel2': 8,
            'Paired': 12,
            'Accent': 8,
            'Dark2': 8,
            'Set1': 9,
            'Set2': 8,
            'Set3': 12,
            'tab10': 10,
            'tab20': 20,
            'tab20b': 20,
            'tab20c': 20}

    qcm_names = list(nr_c.keys())  # valid names of qualititative color maps
    qcm_names_lower = ['pastel1','pastel2','paired','accent','dark2','set1',
                       'set2','set3']  # lower case version (without 'tab' ones)

    if not isinstance(color_scheme,(str,unicode,int,float)):
        raise TypeError('Unrecognizable data type for color_scheme.')

    if color_scheme == 'rgbcmyk':  # default matplotlib v1.5 color scheme
        palette = ['b','g','r','c','m','y','k']
    elif color_scheme in ['bw','bw3']:  # black and white, 3 levels
        palette = [[0]*3,[0.4]*3,[0.75]*3]
    elif color_scheme == 'bw4':  # black and white, 4 levels
        palette = [[0]*3,[0.25]*3,[0.5]*3,[0.75]*3]
    elif color_scheme == 'bw5':  # black and white, 5 levels
        palette = [[0]*3,[0.15]*3,[0.3]*3,[0.5]*3,[0.7]*3,]
    elif color_scheme == 'tab10':
        ## This is the default color scheme, and it is a duplicate if the next
        ## case. The only difference is that this case returns HEX values instead
        ## of RGB values, hence shorter (better for demonstrative purposes.)
        palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                   '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    elif color_scheme in qcm_names:
        c_s = color_scheme  # short hand [Note: no wrap-around behavior in mpl.cm functions]
        rgba = eval('mpl.cm.%s(range(%d))' % (c_s,nr_c[c_s]))  # e.g., mpl.cm.Set1(range(10))
        palette = [list(_)[:3] for _ in rgba]  # remove alpha value from each sub-list
    elif color_scheme in qcm_names_lower:
        c_s = color_scheme.title()  # first letter upper case
        rgba = eval('mpl.cm.%s(range(%d))' % (c_s,nr_c[c_s]))
        palette = [list(_)[:3] for _ in rgba]
    elif color_scheme == 8.3:  # MATLAB ver 8.3 (R2014a) and earlier
        palette = [[0, 0, 1.0000],  # blue
                   [0, 0.5000, 0],  # green
                   [1.0000, 0, 0],  # red
                   [0, 0.7500, 0.7500],  # cyan
                   [0.7500, 0, 0.7500],  # magenta
                   [0.7500, 0.7500, 0],  # dark yellow
                   [0.2500, 0.2500, 0.2500]]  # dark gray
    elif color_scheme == 8.4:  # MATLAB ver 8.4 (R2014b) and later
        palette = [[0.0000, 0.4470, 0.7410],
                   [0.8500, 0.3250, 0.0980],
                   [0.9290, 0.6940, 0.1250],
                   [0.4940, 0.1840, 0.5560],
                   [0.4660, 0.6740, 0.1880],
                   [0.3010, 0.7450, 0.9330],
                   [0.6350, 0.0780, 0.1840]]
    else:
        raise ValueError('Invalid value of color_scheme.')

    L = len(palette)
    if N is None:
        N = L
    elif not isinstance(N,(int,np.integer)):
        raise TypeError('N should be either None or integers.')
    return [palette[i % L] for i in range(N)]  # wrap around 'palette' if N > L

#%%############################################################################
def get_linespecs(color_scheme='tab10', n_linestyle=4, range_linewidth=[1,2,3],
                  priority='color'):
    '''
    Returns a list of distinguishable line specifications (color, line style,
    and line width combinations).

    Parameters
    ----------
    color_scheme : <str> or {8.3, 8.4}
        Color scheme specifier. See docs of get_colors() for valid specifiers.
    n_linestyle : {1, 2, 3, 4}
        Number of different line styles to use. There are only 4 available line
        stylies in Matplotlib to choose from: (1) - (2) -- (3) -. and (4) ..
        For example, if you use 2, you choose only - and --
    range_linewidth : array_like
        The range of different line width values to use.
    priority : {'color', 'linestyle', 'linewidth'}
        Which one of the three line specification aspects (i.e., color, line
        style, or line width) should change first in the resulting list of
        line specifications.

    Returns
    -------
    A list whose every element is a dictionary that looks like this:
    {'color': '#1f77b4', 'ls': '-', 'lw': 1}. Each element can then be passed
    as keyword arguments to matplotlib.pyplot.plot() or other similar functions.

    Example
    -------
    >>> import plot_utils as pu
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([0,1],[0,1],**pu.get_linespecs()[53])
    '''

    import cycler

    colors = get_colors(N=None,color_scheme=color_scheme)
    if n_linestyle in [1,2,3,4]:
        linestyles = ['-', '--', '-.', ':'][:n_linestyle]
    else:
        raise ValueError('n_linestyle should be 1, 2, 3, or 4.')

    color_cycle = cycler.cycler(color=colors)
    ls_cycle = cycler.cycler(ls=linestyles)
    lw_cycle = cycler.cycler(lw=range_linewidth)

    if priority == 'color':
        style_cycle = lw_cycle * ls_cycle * color_cycle
    elif priority == 'linestyle':
        style_cycle = lw_cycle * color_cycle * ls_cycle
    elif priority == 'linewidth':
        style_cycle = color_cycle * lw_cycle * ls_cycle

    return list(style_cycle)

#%%############################################################################
def linespecs_demo(line_specs, horizontal_plot=False):
    '''
    Demonstrate all generated line specifications given by get_linespecs()

    Parameter
    ---------
    line_spec :
        A list of dictionaries that is the returned value of get_linespecs().
    horizontal_plot : <bool>
        Whether or not to demonstrate the line specifications in a horizontal
        plot.

    Returns
    -------
    fig, ax :
        Figure and axes objects.
    '''
    x = np.arange(0,10,0.05)  # define x and y points to plot
    y = np.sin(x)
    fig_width = 8
    fig_height = len(line_specs) * 0.2
    if horizontal_plot:
        x, y = y, x
        fig_width, fig_height = fig_height, fig_width

    figsize = (fig_width, fig_height)
    fig = plt.figure(figsize=figsize)
    ax =  plt.axes()

    for j, linespec in enumerate(line_specs):
        if horizontal_plot:
            plt.plot(x+j, y, **linespec)
        else:
            plt.plot(x, y-j, **linespec)

    ax.axis('off')  # no coordinate axes box

    return fig, ax

#%%############################################################################
def find_axes_lim(data_limit,tick_base_unit,direction='upper'):
    '''
    Return a "whole" number to be used as the upper or lower limit of axes.

    For example, if the maximum x value of the data is 921.5, and you would
    like the upper x_limit to be a multiple of 50, then this function returns
    950.

    Parameters
    ----------
    data_limit: The upper and/or lower limit(s) of data.
                (1) If a tuple (or list) of two elements is provided, then
                    the upper and lower axis limits are automatically
                    determined. (The order of the two elements does not
                    matter.)
                (2) If a scalar (float or int)is provided, then the axis
                    limit is determined based on the DIRECTION provided.
    tick_base_unit: For example, if you want your axis limit(s) to be a
                    multiple of 20 (such as 80, 120, 2020, etc.), then use
                    20.
    direction: 'upper' or 'lower'; used only when data_limit is a scalar.
               If data_limit is a tuple/list, then this variable is
               disregarded.

    Returns
    -------
    If data_limit is a list/tuple of length 2, return [min_limit,max_limit]
    (Note: it is always ordered no matter what the order of data_limit is.)

    If data_limit is a scalar, return axis_limit according to the DIRECTION.
    '''
    if isinstance(data_limit,(float,int)):  # is scalar
        if direction == 'upper':
            return tick_base_unit * (int(data_limit/tick_base_unit)+1)
        elif direction == 'lower':
            return tick_base_unit * (int(data_limit/tick_base_unit))
        else:
            raise ValueError('Length of data_limit should be at least 1.')
    elif isinstance(data_limit,list):
        if len(data_limit) > 2:
            raise ValueError('Length of data_limit should be at most 2.')
        elif len(list(data_limit)) == 2:
            min_data = min(data_limit)
            max_data = max(data_limit)
            max_limit = tick_base_unit * (int(max_data/tick_base_unit)+1)
            min_limit = tick_base_unit * (int(min_data/tick_base_unit))
            return [min_limit, max_limit]
        elif len(data_limit) == 1:  # such as [2.14]
            return find_axes_lim(data_limit[0],tick_base_unit,direction) # recursion
    elif isinstance(data_limit,np.ndarray):
        data_limit = data_limit.flatten()  # convert np.array(2.5) into np.array([2.5])
        if data_limit.size == 1:
            return find_axes_lim(data_limit[0],tick_base_unit,direction) # recursion
        elif data_limit.size == 2:
            return find_axes_lim(list(data_limit),tick_base_unit,direction) # recursion
        elif data_limit.size >= 3:
            raise ValueError('Length of data_limit should be at most 2.')
        else:
            raise TypeError('Unrecognized data type for data_limit.')
    else:
        raise TypeError('Unrecognized data type for data_limit.')

#%%############################################################################
def discrete_histogram(x, fig=None, ax=None, color=None, alpha=None,
                       rot=0, logy=False, title='', figsize=(5,3), dpi=100):
    '''
    Plot a discrete histogram based on "x", such as below:


      N ^
        |
        |           ____
        |           |  |   ____
        |           |  |   |  |
        |    ____   |  |   |  |
        |    |  |   |  |   |  |
        |    |  |   |  |   |  |   ____
        |    |  |   |  |   |  |   |  |
        |    |  |   |  |   |  |   |  |
       -|--------------------------------------->  x
              x1     x2     x3     x4    ...

    In the figure, N is the number of values where x = x1, x2, x3, or x4.
    And x1, x2, x3, x4, etc. are the discrete values within x.

    Parameters
    ----------
    x : <array_like>
        Data to be visualized.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    color : <str> or <list>
        Color of bar. If not specified, the default color (muted blue)
        is used.
    alpha : scalar
        Opacity of bar. If not specified, the default value (1.0) is used.
    rot : scalar
        Rotation angle (degrees) of x axis label. Default = 0 (upright label)

    Returns
    -------
    fig, ax :
        Figure and axes objects

    Reference
    ---------
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html
    http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#bar-plots
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    X = pd.Series(x)  # convert x into series
    value_count = X.value_counts().sort_index()  # count distinct values and sort
    ax = value_count.plot.bar(color=color,alpha=alpha,ax=ax,rot=rot)#,logy=logy,ylim=(1,10000))
    ax.set_ylabel('Number of occurrences')
    if np.abs(rot) > 0:
        ax.set_xticklabels(value_count.index,rotation=rot,ha='right')
    if logy:   # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.yscale
        ax.set_yscale('log', nonposy='clip')  # https://stackoverflow.com/a/17952890
    ax.set_title(title)

    return fig, ax

#%%############################################################################
from matplotlib.ticker import ScalarFormatter
class FixedOrderFormatter(ScalarFormatter):
    '''
    Formats axis ticks using scientific notation with a constant order of
    magnitude.

    (Reference: https://stackoverflow.com/a/3679918)
    '''

    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

#%%############################################################################
def choropleth_map_state(data_per_state, figsize=(10,7),
                         dpi=100, vmin=None, vmax=None, map_title='USA map',
                         unit='', cmap='OrRd', fontsize=14, cmap_midpoint=None,
                         shapefile_dir=None):
    '''
    Generate a choropleth map of USA (including Alaska and Hawaii), on a state
    level.

    According to wikipedia, a choropleth map is a thematic map in which areas
    are shaded or patterned in proportion to the measurement of the statistical
    variable being displayed on the map, such as population density or
    per-capita income.

    Parameters
    ----------
    data_per_state : <dict> or <pd.Series> or <pd.DataFrame>
        Numerical data of each state, to be plotted onto the map.
        Acceptable data types include:
            - pandas Series: Index should be valid state identifiers (i.e.,
                             state full name, abbreviation, or FIPS code)
            - pandas DataFrame: The dataframe can have only one columns (with
                                the index being valid state identifiers), two
                                columns (with one of the column named 'state',
                                'State', or 'FIPS_code', and containing state
                                identifiers).
            - dictionary: with keys being valid state identifiers, and values
                          being the numerical values to be visualized
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    vmin : scalar
        Minimum value to be shown on the map. If vmin is larger than the
        actual minimum value in the data, some of the data values will be
        "clipped". This is useful if there are extreme values in the data
        and you do not want those values to complete skew the color
        distribution.
    vmax : scalar
        Maximum value to be shown on the map. Similar to vmin.
    map_title : <str>
        Title of the map, to be shown on the top of the map.
    unit : <str>
        Unit of the numerical (for example, "population per km^2"), to be
        shown on the right side of the color bar.
    cmap : <str> or <matplotlib.colors.Colormap>
        Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
        for plotting deviation maps.
    fontsize : scalar
        Font size of all the texts on the map.
    cmap_midpoint : scalar
        A numerical value that specifies the "deviation point". For example,
        if your data ranges from -200 to 1000, and you want negative values
        to appear blue-ish, and positive values to appear red-ish, then you
        can set cmap_midpoint to 0.0.
    shapefile_dir : <str>
        Directory where shape files are stored. Shape files (state level and
        county level) should be organized as follows:
            [shapefile_dir]/usa_states/st99_d00.(...)
            [shapefile_dir]/usa_counties/cb_2016_us_county_500k.(...)

    Returns
    -------
    fig, ax :
        Figure and axes objects

    References
    ----------
        I based my modifications partly on some code snippets in this
        stackoverflow thread: https://stackoverflow.com/questions/39742305
    '''

    from mpl_toolkits.basemap import Basemap as Basemap
    from matplotlib.colors import rgb2hex, Normalize
    from matplotlib.patches import Polygon
    from matplotlib.colorbar import ColorbarBase

    if isinstance(data_per_state,pd.Series):
        data_per_state = data_per_state.to_dict()  # convert to dict
    elif isinstance(data_per_state,pd.DataFrame)  and data_per_state.shape[1] == 1:
        data_per_state = data_per_state.iloc[:,0].to_dict()
    elif isinstance(data_per_state,pd.DataFrame) and data_per_state.shape[1] == 2:
        if 'FIPS_code' in data_per_state.columns:
            data_per_state = data_per_state.set_index('FIPS_code')
        elif 'state' in data_per_state.columns:
            data_per_state = data_per_state.set_index('state')
        elif 'State' in data_per_state.columns:
            data_per_state = data_per_state.set_index('State')
        else:
            sys.exit('------  Input data format not recognized!   ----------')
        data_per_state = data_per_state.iloc[:,0].to_dict()

    #  if dict keys are state abbreviations such as "AK", "CA", etc.
    if len(list(data_per_state.keys())[0])==2 and list(data_per_state.keys())[0].isalpha():
        data_per_state = translate_state_abbrev(data_per_state) # convert from 'AK' to 'Alaska'

    #  if dict keys are state FIPS codes such as "01", "45", etc.
    if len(list(data_per_state.keys())[0])==2 and list(data_per_state.keys())[0].isdigit():
        data_per_state = convert_FIPS_to_state_name(data_per_state) # convert from '01' to 'Alabama'

    data_per_state = check_all_states(data_per_state)  # see function definition of check_all_states()

    fig, ax = process_fig_ax_objects(None, None, figsize, dpi)

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(llcrnrlon=-190,llcrnrlat=20,urcrnrlon=-143,urcrnrlat=46,
                projection='merc',lat_ts=20)  # do not change these numbers

    #---------   draw state boundaries  ----------------------------------------
    if shapefile_dir is None:
        shapefile_dir = './shapefiles'
    shp_path_state = os.path.join(shapefile_dir,'usa_states','st99_d00')
    try:
        shp_info = m.readshapefile(shp_path_state,'states',drawbounds=True,
                                   linewidth=0.45,color='gray')
        shp_info_ = m_.readshapefile(shp_path_state,'states',drawbounds=False)
    except:
        raise IOError('Shape files not found. Specify the location of the "shapefiles" folder.')

    #-------- choose a color for each state based on population density. -------
    colors={}
    statenames=[]
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(list(data_per_state.values()))
    if vmax is None:
        vmax = np.nanmax(list(data_per_state.values()))
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            data_ = data_per_state[statename]
            if not np.isnan(data_):
                # calling colormap with value between 0 and 1 returns rgba value.
                colors[statename] = cmap(float(data_-vmin)/(vmax-vmin))[:3]
            else:  # if data_ is NaN, set color to light grey, and with hatching pattern
                colors[statename] = None #np.nan#[0.93]*3
        statenames.append(statename)

    #---------  cycle through state names, color each one.  --------------------
    ax = plt.gca() # get current axes instance

    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            if colors[statenames[nshape]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=[0.4]*3,hatch='\\')
            else:
                color = rgb2hex(colors[statenames[nshape]])
                poly = Polygon(seg,facecolor=color,edgecolor=color)

            ax.add_patch(poly)

    AREA_1 = 0.005  # exclude small Hawaiian islands that are smaller than AREA_1
    AREA_2 = AREA_1 * 30.0  # exclude Alaskan islands that are smaller than AREA_2
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for nshape, shapedict in enumerate(m_.states_info):  # plot Alaska and Hawaii as map insets
        if shapedict['NAME'] in ['Alaska', 'Hawaii']:
            seg = m_.states[int(shapedict['SHAPENUM'] - 1)]
            if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > AREA_1:
                seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            elif shapedict['NAME'] == 'Alaska' and float(shapedict['AREA']) > AREA_2:
                seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y)\
                       for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor='gray', linewidth=.45)
            ax.add_patch(poly)

    ax.set_title(map_title)

    #---------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8]*3  # define light gray color RGB
    m_.plot(np.linspace(170,177),np.linspace(29,29),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(177,180),np.linspace(29,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(180,180),np.linspace(26,23),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-177),np.linspace(23,20),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-175),np.linspace(26,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-175,-171),np.linspace(26,22),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-171,-171),np.linspace(22,20),linewidth=1.,
            color=light_gray,latlon=True)

    #---------   Show color bar  ---------------------------------------
    if cmap_midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=cmap_midpoint)

    ax_c = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical',label=unit)

    mpl_version = mpl.__version__.split('.')
    if float(mpl_version[0] + '.' + mpl_version[0]) >= 2.1: # version > 2.1.0
        cb = adjust_colorbar_tick_labels(cb,
                                         np.nanmax(list(data_per_state.values())) > vmax,
                                         np.nanmin(list(data_per_state.values())) < vmin)

    #---------   Set overall font size  --------------------------------
    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax  # return figure and axes handles

#%%############################################################################
def choropleth_map_county(data_per_county, figsize=(10,7),
                          dpi=100, vmin=None, vmax=None, unit='', cmap='OrRd',
                          map_title='USA county map', fontsize=14,
                          cmap_midpoint=None, shapefile_dir=None):
    '''
    Generate a choropleth map of USA (including Alaska and Hawaii), on a county
    level.

    According to wikipedia, a choropleth map is a thematic map in which areas
    are shaded or patterned in proportion to the measurement of the statistical
    variable being displayed on the map, such as population density or
    per-capita income.

    Parameters
    ----------
    data_per_county : <dict> or <pd.Series> or <pd.DataFrame>
        Numerical data of each county, to be plotted onto the map.
        Acceptable data types include:
            - pandas Series: Index should be valid county identifiers (i.e.,
                             5 digit county FIPS codes)
            - pandas DataFrame: The dataframe can have only one columns (with
                                the index being valid county identifiers), two
                                columns (with one of the column named 'state',
                                'State', or 'FIPS_code', and containing county
                                identifiers).
            - dictionary: with keys being valid county identifiers, and values
                          being the numerical values to be visualized
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    vmin : scalar
        Minimum value to be shown on the map. If vmin is larger than the
        actual minimum value in the data, some of the data values will be
        "clipped". This is useful if there are extreme values in the data
        and you do not want those values to complete skew the color
        distribution.
    vmax : scalar
        Maximum value to be shown on the map. Similar to vmin.
    map_title : <str>
        Title of the map, to be shown on the top of the map.
    unit : <str>
        Unit of the numerical (for example, "population per km^2"), to be
        shown on the right side of the color bar.
    cmap : <str> or <matplotlib.colors.Colormap>
        Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
        for plotting deviation maps.
    fontsize : scalar
        Font size of all the texts on the map.
    cmap_midpoint : scalar
        A numerical value that specifies the "deviation point". For example,
        if your data ranges from -200 to 1000, and you want negative values
        to appear blue-ish, and positive values to appear red-ish, then you
        can set cmap_midpoint to 0.0.
    shapefile_dir : <str>
        Directory where shape files are stored. Shape files (state level and
        county level) should be organized as follows:
            [shapefile_dir]/usa_states/st99_d00.(...)
            [shapefile_dir]/usa_counties/cb_2016_us_county_500k.(...)

    Returns
    -------
    fig, ax :
        Figure and axes objects

    References
    ----------
        I based my modifications partly on some code snippets in this
        stackoverflow thread: https://stackoverflow.com/questions/39742305
    '''

    from mpl_toolkits.basemap import Basemap as Basemap
    from matplotlib.colors import rgb2hex, Normalize
    from matplotlib.patches import Polygon
    from matplotlib.colorbar import ColorbarBase

    if isinstance(data_per_county,pd.Series):
        data_per_county = data_per_county.to_dict()  # convert to dict
    elif isinstance(data_per_county,pd.DataFrame) and data_per_county.shape[1] == 1:
        data_per_county = data_per_county.iloc[:,0].to_dict()
    elif isinstance(data_per_county,pd.DataFrame) and data_per_county.shape[1] == 2:
        data_per_county = data_per_county.set_index('FIPS_code')  # colunm name hard-coded here for safety
        data_per_county = data_per_county.iloc[:,0].to_dict()

    fig, ax = process_fig_ax_objects(None, None, figsize, dpi)

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(llcrnrlon=-190,llcrnrlat=20,urcrnrlon=-143,urcrnrlat=46,
                projection='merc',lat_ts=20)  # do not change these numbers

    #---------   draw state and county boundaries  ----------------------------
    if shapefile_dir is None:
        shapefile_dir = './shapefiles'
    shp_path_state = os.path.join(shapefile_dir,'usa_states','st99_d00')
    try:
        shp_info = m.readshapefile(shp_path_state,'states',drawbounds=True,
                                   linewidth=0.45,color='gray')
        shp_info_ = m_.readshapefile(shp_path_state,'states',drawbounds=False)
    except:
        raise IOError('Shape files not found. Specify the location of the "shapefiles" folder.')

    cbc = [0.75]*3  # county boundary color
    cbw = 0.15  # county boundary line width
    shp_path_county = os.path.join(shapefile_dir,'usa_counties','cb_2016_us_county_500k')
    try:
        shp_info_cnty = m.readshapefile(shp_path_county,'counties',drawbounds=True,
                                        linewidth=cbw,color=cbc)

        shp_info_cnty_ = m_.readshapefile(shp_path_county,'counties',drawbounds=False)
    except:
        raise IOError('Shape files not found. Specify the location of the "shapefiles" folder.')

    #-------- choose a color for each county based on unemployment rate -------
    colors={}
    county_FIPS_code_list=[]
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(list(data_per_county.values()))
    if vmax is None:
        vmax = np.nanmax(list(data_per_county.values()))
    for shapedict in m.counties_info:
        county_FIPS_code = shapedict['GEOID']
        if county_FIPS_code in data_per_county.keys():
            data_ = data_per_county[county_FIPS_code]
        else:
            data_ = np.nan

        # calling colormap with value between 0 and 1 returns rgba value.
        if not np.isnan(data_):
            colors[county_FIPS_code] = cmap(float(data_-vmin)/(vmax-vmin))[:3]
        else:
            colors[county_FIPS_code] = None

        county_FIPS_code_list.append(county_FIPS_code)

    #---------  cycle through county names, color each one.  --------------------
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for j, seg in enumerate(m.counties):  # for 48 lower states
        shapedict = m.counties_info[j]  # query shape dict at j-th position
        if shapedict['STATEFP'] not in ['02','15']:  # not Alaska or Hawaii
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=color)#,hatch='\\')
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)

    for j, seg in enumerate(m_.counties):  # for Alaska and Hawaii
        shapedict = m.counties_info[j]  # query shape dict at j-th position
        if shapedict['STATEFP']=='02':  # Alaska
            seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y) for x,y in seg]
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)#,hatch='\\')
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)
            ax.add_patch(poly)
        if shapedict['STATEFP']=='15':  # Hawaii
            seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)#,hatch='\\')
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)
            ax.add_patch(poly)

    ax.set_title(map_title)

    #------------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8]*3  # define light gray color RGB
    m_.plot(np.linspace(170,177),np.linspace(29,29),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(177,180),np.linspace(29,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(180,180),np.linspace(26,23),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-177),np.linspace(23,20),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-175),np.linspace(26,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-175,-171),np.linspace(26,22),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-171,-171),np.linspace(22,20),linewidth=1.,
            color=light_gray,latlon=True)

    #------------   Show color bar   ---------------------------------------
    if cmap_midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=cmap_midpoint)

    ax_c = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical',label=unit)

    mpl_version = mpl.__version__.split('.')
    if float(mpl_version[0] + '.' + mpl_version[0]) >= 2.1: # version > 2.1.0
        cb = adjust_colorbar_tick_labels(cb,
                                         np.nanmax(list(data_per_county.values())) > vmax,
                                         np.nanmin(list(data_per_county.values())) < vmin)

    #------------   Set overall font size  --------------------------------
    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax  # return figure and axes handles

#%%############################################################################
def adjust_colorbar_tick_labels(colorbar_obj,adjust_top=True,adjust_bottom=True):
    '''
    Given a colorbar object (colorbar_obj), change the text of the top (and/or
    bottom) tick label text.

    For example, the top tick label of the color bar is originally "1000", then
    this function change it to ">1000", to represent the cases where the colors
    limits are manually clipped at a certain level (useful for cases with
    extreme values in only some limited locations in the color map).

    Similarly, this function adjusts the lower limit.
    For example, the bottom tick label is originally "0", then this function
    changes it to "<0".

    The second and third parameters control whether or not this function adjusts
    top/bottom labels, and which one(s) to adjust.

    Note: get_ticks() only exists in matplotlib version 2.1.0+, and this function
          does not check for matplotlib version. Use with caution.
    '''

    cbar_ticks = colorbar_obj.get_ticks()  # get_ticks() is only added in ver 2.1.0
    new_ticks = [str(int(a)) if int(a)==a else str(a) for a in cbar_ticks]  # convert to int if possible

    if (adjust_top == True) and (adjust_bottom == True):
        new_ticks[-1] = '>' + new_ticks[-1]   # adjust_top and adjust_bottom may
        new_ticks[0] = '<' + new_ticks[0]     # be numpy.bool_ type, which is
    elif adjust_top == True:                  # different from Python bool type!
        new_ticks[-1] = '>' + new_ticks[-1]   # Thus 'adjust_top == True' is used
    elif adjust_bottom == True:               # here, instead of 'adjust_top is True'.
        new_ticks[0] = '<' + new_ticks[0]
    else:
        pass

    colorbar_obj.ax.set_yticklabels(new_ticks)

    return colorbar_obj

#%%============================================================================
class MidpointNormalize(Normalize):
    '''
    Auxiliary class definition. Copied from:
    https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989#20146989
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#%%============================================================================
def convert_FIPS_to_state_name(dict1):
    '''
    Convert state FIPS codes such as '01' and '45' into full state names.
    '''
    fips2state = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", \
              "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", \
              "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", \
              "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", \
              "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", \
              "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", \
              "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", \
              "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", \
              "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", \
              "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", \
              "56": "WY"}  # dictionary mapping FIPS code to state abbreviation

    dict2 = {}  # create empty dict
    for FIPS_code in dict1.keys():
        new_state_name = fips2state[FIPS_code]  # convert state name
        dict2.update({new_state_name: dict1[FIPS_code]})

    dict3 = translate_state_abbrev(dict2,abbrev_to_full=True)

    return dict3

#%%============================================================================
def translate_state_abbrev(dict1,abbrev_to_full=True):
    '''
    Convert state full names into state abbreviations, or the other way.
    '''
    if abbrev_to_full is True:
        translation = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
        }
    else:
        translation = {
            'Alabama': 'AL',
            'Alaska': 'AK',
            'Arizona': 'AZ',
            'Arkansas': 'AR',
            'California': 'CA',
            'Colorado': 'CO',
            'Connecticut': 'CT',
            'Delaware': 'DE',
            'Florida': 'FL',
            'Georgia': 'GA',
            'Hawaii': 'HI',
            'Idaho': 'ID',
            'Illinois': 'IL',
            'Indiana': 'IN',
            'Iowa': 'IA',
            'Kansas': 'KS',
            'Kentucky': 'KY',
            'Louisiana': 'LA',
            'Maine': 'ME',
            'Maryland': 'MD',
            'Massachusetts': 'MA',
            'Michigan': 'MI',
            'Minnesota': 'MN',
            'Mississippi': 'MS',
            'Missouri': 'MO',
            'Montana': 'MT',
            'Nebraska': 'NE',
            'Nevada': 'NV',
            'New Hampshire': 'NH',
            'New Jersey': 'NJ',
            'New Mexico': 'NM',
            'New York': 'NY',
            'North Carolina': 'NC',
            'North Dakota': 'ND',
            'Ohio': 'OH',
            'Oklahoma': 'OK',
            'Oregon': 'OR',
            'Pennsylvania': 'PA',
            'Rhode Island': 'RI',
            'South Carolina': 'SC',
            'South Dakota': 'SD',
            'Tennessee': 'TN',
            'Texas': 'TX',
            'Utah': 'UT',
            'Vermont': 'VT',
            'Virginia': 'VA',
            'Washington': 'WA',
            'West Virginia': 'WV',
            'Wisconsin': 'WI',
            'Wyoming': 'WY',
        }

    dict2 = {}  # create empty dict
    for state_name in dict1.keys():
        new_state_name = translation[state_name]  # convert state name
        dict2.update({new_state_name: dict1[state_name]})

    return dict2

#%%============================================================================
def check_all_states(dict1):
    '''
    Check whether dict1 has all 50 states of USA. If not, append missing state(s)
    to the dictionary and assign np.nan value as its value.

    The state names of dict1 must be full names.
    '''
    full_state_list = [

         'Alabama','Alaska','Arizona','Arkansas','California','Colorado',
         'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho',
         'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
         'Maine', 'Maryland','Massachusetts','Michigan','Minnesota',
         'Mississippi', 'Missouri','Montana','Nebraska','Nevada',
         'New Hampshire','New Jersey','New Mexico','New York',
         'North Carolina','North Dakota','Ohio',
         'Oklahoma','Oregon','Pennsylvania','Rhode Island',
         'South Carolina','South Dakota','Tennessee','Texas','Utah',
         'Vermont','Virginia','Washington','West Virginia',
         'Wisconsin','Wyoming'
    ]

    if list(dict1.keys()).sort() != full_state_list:
        dict2 = {}  # create new list
        for state in full_state_list:
            if state in dict1.keys():
                dict2[state] = dict1[state]
            else:
                print('%s data missing (replaced with NaN).'%state)
                dict2[state] = np.nan
    else:
        dict2 = dict1

    return dict2

#%%============================================================================
def plot_timeseries(time_series, date_fmt=None, fig=None, ax=None, figsize=(10,3),
                    dpi=100, xlabel='Time', ylabel=None, label=None, color=None,
                    lw=2, ls=None, marker=None, fontsize=12, xgrid_on=True,
                    ygrid_on=True, title=None, zorder=None,
                    month_grid_width=None):
    '''
    Plot time_series, where its index indicates dates (e.g., year, month, date).

    You can plot multiple time series by supplying a multi-column pandas
    Dataframe as time_series, but you cannot use custom line specifications
    (colors, width, and styles) for each time series. It is recommended to use
    plot_multiple_timeseries() in stead.

    Parameters
    ----------
    time_series : <pd.Series> or <pd.DataFrame>
        A pandas Series, with index being date; or a pandas DataFrame, with
        index being date, and each column being a different time series.
    date_fmt : <str>
        Date format specifier, e.g., '%Y-%m' or '%d/%m/%y'.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        figure size (width, height) in inches (fig object passed via
        "fig" will over override this parameter)
    dpi : scalar
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    xlabel : <str>
        Label of X axis. Usually "Time" or "Date"
    ylabel : <str>
        Label of Y axis. Usually the meaning of the data
    label : <str>
        Label of data, for plotting legends
    color : <list> or <str>
        Color of line. If None, let Python decide for itself.
    xgrid_on : <bool>
        Whether or not to show vertical grid lines (default: True)
    ygrid_on : <bool>
        Whether or not to show horizontal grid lines (default: True)
    title : <str>
        Figure title (optional)
    zorder : scalar
        Set the zorder for lines. Higher zorder are drawn on top.
    month_grid_width : <scalar>
        the on-figure "horizontal width" that each time interval occupies.
        This value determines how X axis labels are displayed (e.g., smaller
        width leads to date labels being displayed with 90 deg rotation).
        Do not change this unless you really know what you are doing.

    Returns
    -------
    fig, ax :
        Figure and axes objects
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    ax_size = get_ax_size(fig, ax)

    ts = time_series.copy()  # shorten the name + avoid changing some_time_series
    ts.index = as_date(ts.index, date_fmt)  # batch-convert index to Timestamp format of pandas

    if zorder:
        ax.plot(ts.index,ts,color=color,lw=lw,ls=ls,marker=marker,label=label,zorder=zorder)
    else:
        ax.plot(ts.index,ts,color=color,lw=lw,ls=ls,marker=marker,label=label)
    ax.set_label(label)  # set label for legends using argument 'label'
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if month_grid_width == None:  # width of each month in inches
        month_grid_width = float(ax_size[0])/calc_month_interval(ts.index)
    ax = format_xlabel(ax,month_grid_width)

    if ygrid_on == True:
        ax.yaxis.grid(ls=':',color=[0.75]*3)
    if xgrid_on == True:
        ax.xaxis.grid(False,'major')
        ax.xaxis.grid(xgrid_on,'minor',ls=':',color=[0.75]*3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax

#%%============================================================================
def plot_multiple_timeseries(multiple_time_series, show_legend=True,
                             fig=None, ax=None, figsize=(10,3), dpi=100,
                             ncol_legend=5, **kwargs):
    '''
    Plot multiple_time_series, where its index indicates dates (e.g., year,
    month, date).

    Note that setting keyword arguments such as color or ls ("linestyle") will
    force all time series to have the same color or ls. So it is recommended
    to let this function generate distinguishable line specifications (color/
    linestyle/linewidth combinations) by itself. (Although the more time series,
    the less the distinguishability. 240 time series or less is recommended.)

    Parameters
    ----------
    multiple_time_series : <pandas.DataFrame>
        A pandas dataframe, with index being date , andeach column being a
        different time series.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Figure size (width, height) in inches (fig object passed via
        "fig" will over override this parameter)
    dpi : <scalar>
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    ncol_legend : <int>
        Number of columns of the legend
    **kwargs :
        Other keyword arguments to be passed to plot_timeseries(), such as
        color, marker, fontsize, etc. (Check docstring of plot_timeseries()).

    Returns
    -------
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if show_legend is False:  # if no need to show legends, just pass everything
        fig, ax = plot_timeseries(multiple_time_series, fig, ax, dpi, **kwargs)
    else:
        if isinstance(multiple_time_series,pd.Series):
            nr_timeseries  = 1
            multiple_time_series = pd.DataFrame(multiple_time_series,copy=True)
        else:
            nr_timeseries = multiple_time_series.shape[1]

        if nr_timeseries <= 40:  # 10 colors x 4 linestyles = 40, so use lw=2
            linespecs = get_linespecs(range_linewidth=[2])
        elif nr_timeseries <= 120:  # need multiple line widths
            linespecs = get_linespecs(range_linewidth=[1,3,5])
        elif nr_timeseries <= 240:
            linespecs = get_linespecs(color_scheme='tab20',range_linewidth=[1,3,5])
        else:
            linespecs = get_linespecs(color_scheme='tab20',  # use more line widths
                            range_linewidth=range(1,(nr_timeseries-1)//240+5,2))

        for j in range(nr_timeseries):
            tmp_dict = linespecs[j % nr_timeseries].copy()
            tmp_dict.update(kwargs)  # kwargs overwrites tmp_dict if key already exists in tmp_dict
            if 'lw' in tmp_dict.keys():  # thinner lines above thicker lines
                zorder = 1 + 1.0/tmp_dict['lw']  # and "+1" to put all lines above grid line

            plot_timeseries(multiple_time_series.iloc[:,j],
                            fig=fig, ax=ax, label=multiple_time_series.columns[j],
                            zorder=zorder, **tmp_dict)

        if 'title' not in kwargs:
            bbox_anchor_loc = (0., 1.02, 1., .102)
        else:
            bbox_anchor_loc = (0., 1.08, 1., .102)
        ax.legend(bbox_to_anchor=bbox_anchor_loc,
                  loc='lower center', ncol=ncol_legend)

    ax.set_axisbelow(True)
    return fig, ax

#%%============================================================================
def fill_timeseries(time_series, upper_bound, lower_bound, date_fmt=None,
                    fig=None, ax=None, figsize=(10,3), dpi=100,
                    xlabel='Time', ylabel=None, label=None,
                    color=None, lw=3, ls='-', fontsize=12, title=None,
                    xgrid_on=True, ygrid_on=True):
    '''
    Plot time_series as a line, where its index indicates a date (e.g., year,
    month, date).

    And then plot the upper bound and lower bound as shaded areas beneath the line.

    Parameters
    ----------
    time_series : <pd.Series> or <pd.DataFrame>
        a pandas Series, with index being date
    upper_bound, lower_bound : <pd.Series>
        upper/lower bounds of the time series, must have the same length as
        time_series
    date_fmt : <str>
        Date format specifier, e.g., '%Y-%m' or '%d/%m/%y'.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        figure size (width, height) in inches (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    xlabel : <str>
        Label of X axis. Usually "Time" or "Date"
    ylabel : <str>
        Label of Y axis. Usually the meaning of the data
    label : <str>
        Label of data, for plotting legends
    color : <str> or list or tuple
        Color of line. If None, let Python decide for itself.
    lw : scalar
        line width of the line that represents time_series
    ls : <str>
        line style of the line that represents time_series
    fontsize : scalar
        font size of the texts in the figure
    title : <str>
        Figure title (optional)
    xgrid_on : <bool>
        Whether or not to show vertical grid lines (default: True)
    ygrid_on : <bool>
        Whether or not to show horizontal grid lines (default: True)

    Returns
    -------
    fig, ax :
        Figure and axes objects
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    ts = time_series.copy()  # shorten the name + avoid changing some_time_series
    ts.index = as_date(ts.index, date_fmt)  # batch-convert index to Timestamp format of pandas
    lb = lower_bound.copy()
    ub = upper_bound.copy()

    ax.fill_between(ts.index,lb,ub,color=color,facecolor=color,
                    linewidth=0.01,alpha=0.5,interpolate=True)
    ax.plot(ts.index,ts,color=color,lw=lw,ls=ls,label=label)
    ax.set_label(label)  # set label for legends using argument 'label'
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    month_grid_width = float(figsize[0])/calc_month_interval(ts.index) # width of each month in inches
    ax = format_xlabel(ax,month_grid_width)

    if ygrid_on == True:
        ax.yaxis.grid(ygrid_on,ls=':',color=[0.75]*3)
    if xgrid_on == True:
        ax.xaxis.grid(False,'major')
        ax.xaxis.grid(xgrid_on,'minor',ls=':',color=[0.75]*3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax

#%%============================================================================
def calc_month_interval(date_array):
    '''
    Calculate how many months are there between the first month and the last
    month of the given date_array.
    '''

    date9 = list(date_array)[-1]
    date0 = list(date_array)[0]
    delta_days = (date9 - date0).days
    if delta_days < 30:  # within one month
        delta_months = delta_days/30.0  # return a float between 0 and 1
    else:
        delta_months = delta_days//30
    return delta_months

#%%============================================================================
def calc_bar_width(width):
    '''
    Calculate width (in points) of bar plot from figure width (in inches)
    '''

    if width <= 7:
        bar_width = width * 3.35  # these numbers are manually fine-tuned
    elif width <= 9:
        bar_width = width * 2.60
    elif width <= 10:
        bar_width = width * 2.10
    else:
        bar_width = width * 1.2

    return bar_width

#%%============================================================================
def get_ax_size(fig, ax, unit='inches'):
    '''
    Get size of axes within a figure, given fig and ax objects.

    https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
    '''

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    if unit == 'pixels':
        width *= fig.dpi  # convert from inches to pixels
        height *= fig.dpi
    return width, height

#%%============================================================================
def format_xlabel(ax,month_width):
    '''
    Format the X axis label (which represents dates) in accordance to the width
    of each time interval (month or day).

    For narrower cases, year will be put below month.
    For even narrower cases, not every month will be displayed as a label.
    For very narrow cases (e.g., many years), months will not be displayed, and
    sometimes not every year will be displayed.
    '''

    rot = None  # degree of rotation
    y_int = None  # year interval
    d_int = None  # day interval
    if month_width < 0.038:
        m_int = None  # month interval
        y_int = 3 * int(0.038/month_width)  # interval increases with narrower size
    elif month_width < 0.043:
        m_int = None
        y_int = 2
    elif month_width < 0.05:
        m_int = None
        rot = 30
    elif month_width < 0.055:
        m_int = 6  # display month label for every 6 months
        rot = 30
    elif month_width < 0.06:
        m_int = 6
    elif month_width < 0.09:
        m_int = 5
    elif month_width < 0.11:
        m_int = 4
    elif month_width < 0.16:
        m_int = 3
    elif month_width < 0.29:
        m_int = 2
    else:
        m_int = 1
        if month_width < 1.9:
            pass  # d_int is still None
        elif month_width < 2.9:
            d_int = 15
        elif month_width < 3.5:
            d_int = 10
        elif month_width < 4:
            d_int = 9
        elif month_width < 5:
            d_int = 8
        elif month_width < 6:
            d_int = 7
        elif month_width < 7:
            d_int = 6
        elif month_width < 8.5:
            d_int = 5
        elif month_width < 11:
            d_int = 4
        elif month_width < 15:
            d_int = 3
            rot = 30
        elif month_width < 25.5:
            d_int = 2
            rot = 30
        else:
            d_int = 1
            rot = 30

    if y_int:  # show only every 'y_int' years
        years = mpl.dates.YearLocator(base=y_int)
    else:  # show year on January of every year
        years = mpl.dates.YearLocator()

    xlim = ax.get_xlim()  # number of days since 0001/Jan/1-00:00:00 UTC plus one
    xlim_ = [mpl.dates.num2date(i) for i in xlim]  # convert to datetime object
    if xlim_[0].year == xlim_[1].year:  # if date range is within same year
        if xlim_[0].day > 1:  # not first day of month: show year on next month
            years = mpl.dates.YearLocator(base=1,month=xlim_[0].month+1,day=1)
        else:   # first day of month: show year on this month
            years = mpl.dates.YearLocator(base=1,month=xlim_[0].month  ,day=1)

    if not d_int:  # no day labels will be shown
        months_fmt = mpl.dates.DateFormatter('%m')
    else:
        months_fmt = mpl.dates.DateFormatter('%m/%d')

    if m_int:   # show every 'm_int' months
        if d_int:  # day labels will be shown
            months = mpl.dates.DayLocator(interval=d_int)
        else:
            months = mpl.dates.MonthLocator(interval=m_int)
        ax.xaxis.set_minor_locator(months)
        ax.xaxis.set_minor_formatter(months_fmt)
        if d_int and rot:  # days are shown as rotated
            years_fmt = mpl.dates.DateFormatter('\n\n%Y')  # show year on next next line
        else:
            years_fmt = mpl.dates.DateFormatter('\n%Y')  # show year on next line
    else:  # do not show months in x axis label
        years_fmt = mpl.dates.DateFormatter('%Y')  # show year on current line

    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.tick_params(labelright=True)  # also show y axis on right edge of figure

    if rot and d_int:  # days/months are shown as rotated
        plt.setp(ax.xaxis.get_minorticklabels(),rotation=rot)
    elif rot and not d_int:  # only show years as rotated
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=rot)

    return ax

#%%============================================================================
def as_date(raw_date, date_fmt=None):
    '''
    Converts raw_date to datetime array.

    It can handle:
    (A) A list of str, int, or float, such as:
        [1] ['20150101', '20150201', '20160101']
        [2] ['2015-01-01', '2015-02-01', '2016-01-01']
        [3] [201405, 201406, 201407]
        [4] [201405.0, 201406.0, 201407.0]
    (B) A list of just a single element, such as:
        [1] [201405]
        [2] ['2014-05-25']
        [3] [201412.0]
    (C) A single element of: str, int, float, such as:
        [1] 201310
        [2] 201210.0
    (D) A pandas Series, of length 1 or length larger than 1

    Parameters
    ----------
    raw_date : (see above for acceptable formats)
        The raw date information to be processed
    date_fmt : <str>
        The format of each individual date entry, e.g., '%Y-%m-%d' or '%m/%d/%y'.
        To be passed directly to pd.to_datetime()
        (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html)

    Returns
    -------
    date_list :
        A variable with the same structure (list or scaler-like) as raw_date,
        whose contents have the data type "pandas._libs.tslib.Timestamp".

    Reference: https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    '''

    if pd.__version__ == '0.17.1':
        timestamp_type = pd.tslib.Timestamp
    else:
        timestamp_type = pd._libs.tslib.Timestamp

    if isinstance(raw_date,timestamp_type):  # if already a pandas Timestamp obj
        date_list = raw_date  # return raw_date as is
    else:
        # -----------  Convert to list for pd.Series or np.ndarray objects  -------
        if isinstance(raw_date,(pd.Series,np.ndarray,pd.Index)):
            raw_date = list(raw_date)

        # ----------  Element-wise checks and conversion  -------------------------
        if isinstance(raw_date,list):   # if input is a list
            if len(raw_date) == 0:  # empty list
                date_list = None   # return an empty object
            elif len(raw_date) == 1:  # length of string is 1
                date_ = str(int(raw_date[0]))  # simply unpack it and convert to str int
                date_list = pd.to_datetime(date_, format=date_fmt)
            else:  # length is larger than 1
                nr = len(raw_date)
                date_list = [[None]] * nr
                for j in range(nr):  # loop every element in raw_date
                    j_th = raw_date[j]
                    if isinstance(j_th,(str,unicode)) and j_th.isdigit():
                        date_ = str(int(j_th))
                    elif isinstance(j_th,(str,unicode)) and not j_th.isdigit():
                        date_ = j_th
                    elif isinstance(j_th,(int,np.integer,np.float)):
                        date_ = str(int(j_th))  # robustness not guarenteed!
                    else:
                        raise TypeError('Date type of the element(s) in raw_date not recognized.')
                    date_list[j] = pd.to_datetime(date_, format=date_fmt)
        elif type(raw_date) == dt.date:  # if a datetime.date object
            date_list = raw_date  # no need for conversion
        elif isinstance(raw_date,(int,np.integer,np.float)):  # integer or float
            date_ = str(int(raw_date))
            date_list = pd.to_datetime(date_, format=date_fmt)
        elif isinstance(raw_date,(str,unicode)):  # a single string, such as '2015-04'
            date_ = raw_date  # no conversion needed
            date_list = pd.to_datetime(date_, format=date_fmt)
        else:
            raise TypeError('Input data type of raw_date not recognized.')
            print('\ntype(raw_date) is: %s' % type(raw_date))
            try:
                print('Length of raw_date is: %s' % len(raw_date))
            except TypeError:
                print('raw_date has no length.')

    return date_list

#%%============================================================================
def str2date_kernel(date_):
    '''
    Convert date_ into a datetime object. date_ must be a string (not a list of strings).

    Currently accepted date formats:
    (1) Aug-2014
    (2) August 2014
    (3) 201407
    (4) 2016-07
    (5) 2015-02-21
    '''

    day = None
    if ('-' in date_) and (len(date_) == 8):  # for date style 'Aug-2014'
        month, year = date_.split('-')  # split string by character
        month = dt.datetime.strptime(month,'%b').month  # from 'Mar' to '3'
    elif ' ' in date_:  # for date style 'August 2014'
        month, year = date_.split(' ')  # split string by character
        month = dt.datetime.strptime(month,'%B').month  # from 'March' to '3'
        year = int(year)
    elif (len(date_) == 6) and date_.isdigit():  # for cases like '201205'
        year  = int(date_[:4])  # first four characters
        month = int(date_[4:])  # remaining characters
    elif (len(date_) == 7) and (date_[4]=='-') and not date_.isdigit():  # such as '2015-03' [NOT 100% ROBUST!]
        year, month = date_.split('-')
        year = int(year)
        month = int(month)
    elif (len(date_) == 10) and not date_.isdigit():  # such as '2012-02-01' [NOT 100% ROBUST!!]
        year, month, day = date_.split('-')  # split string by character
        year = int(year)
        month = int(month)
        day = int(day)
    elif (len(date_)==6) and (date_[3]=='-') and (date_[:3].isalpha()) \
         and (date_[4:].isdigit()):  # such as 'May-12'
        month, year = date_.split('-')
        month = dt.datetime.strptime(month,'%b').month  # from 'Mar' to '3'
        year = int(year) + 2000  # from '13' to '2013'
    else:
        print('*****  Edge case encountered! (Date format not recognized.)  *****')
        print('\nUser supplied %s, which is not recognized.\n' % date_)

    if day is None:  # if day is not defined in the if statements
        return dt.date(year,month,1)
    else:
        return dt.date(year,month,day)

#%%============================================================================
def plot_with_error_bounds(x, y, upper_bound, lower_bound,
                           fig=None, ax=None, figsize=None, dpi=100,
                           line_color=[0.4]*3, shade_color=[0.7]*3,
                           shade_alpha=0.5, linewidth=2.0, legend_loc='best',
                           line_label='Data', shade_label='$\mathregular{\pm}$STD',
                           logx=False, logy=False, grid_on=True):
    '''
    Plot a graph with one line and its upper and lower bounds, with areas between
    bounds shaded. The effect is similar to this illustration below.


      y ^            ...                         _____________________
        |         ...   ..........              |                     |
        |         .   ______     .              |  ---  Mean value    |
        |      ...   /      \    ..             |  ...  Error bounds  |
        |   ...  ___/        \    ...           |_____________________|
        |  .    /    ...      \    ........
        | .  __/   ...  ....   \________  .
        |  /    ....       ...          \
        | /  ....            .....       \_
        | ...                    ..........
       -|--------------------------------------->  x


    Parameters
    ----------
    x, y: <array_like>
        data points to be plotted as a line (should have the same length)
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    upper_bound, lower_bound : <array_like>
        Upper and lower bounds to be plotted as shaded areas. Should have the
        same length as x and y.
    line_color : <str> or list or tuple
        color of the line of y
    shade_color : <str> or list or tuple
        color of the underlying shades
    shade_alpha : scalar
        transparency of the shades
    linewidth : scalar
        width of the line of y
    legend_loc : <int> or <str>
        location of the legend, to be passed directly to plt.legend()
    line_label : <str>
        label of the line of y, to be used in the legend
    shade_label : <str>
        label of the shades, to be used in the legend
    logx, logy : <bool>
        Whether or not to show x or y axis scales as log
    grid_on : <bool>
        whether or not to show the grids

    Returns
    -------
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    hl1 = ax.fill_between(x, lower_bound, upper_bound,
                           color=shade_color, facecolor=shade_color,
                           linewidth=0.01, alpha=shade_alpha, interpolate=True,
                           label=shade_label)
    hl2, = ax.plot(x, y, color=line_color, linewidth=linewidth, label=line_label)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')

    if grid_on == True:
        ax.grid(ls=':',lw=0.5)
        ax.set_axisbelow(True)

    plt.legend(handles=[hl2,hl1],loc=legend_loc)

    return fig, ax

#%%============================================================================
def plot_correlation(X, color_map='RdBu_r', fig=None, ax=None,
                     figsize=(6,6), dpi=100, variable_names=None,
                     scatter_plots=False, thres=0.7, ncols_scatter_plots=3):
    '''
    Plot correlation matrix of a dataset X, whose columns are different
    variables (or a sample of a certain random variable).

    Parameters
    ----------
    X : <np.ndarray> or <pd.DataFrame>
        The data set.
    color_map : <str> or <matplotlib.colors.Colormap>
        The color scheme to show high, low, negative high correlations. Legit
        names are listed in https://matplotlib.org/users/colormaps.html. Using
        diverging color maps are recommended: PiYG, PRGn, BrBG, PuOr, RdGy,
        RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr, seismic
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    variable_names : list of <str>
        Names of the variables in X. If X is a pandas dataframe, then this
        argument is not need: column names of X is used as variable names. If
        X is a numpy array, and this argument is not provided, then column
        indices are used. The length of variable_names should match the number
        of columns in X; if not, a warning will be thrown (but not error).
    scatter_plots : bool
        Whether or not to show the variable pairs with high correlation.
        Variable pairs whose absolute value of correlation is higher than thres
        will be plotted as scatter plots.
    thres : scalar between 0 and 1
        Threshold of correlation (absolute value). Variable pairs whose absolute
        correlation is higher than thres will be plotted as scatter plots.
    ncols_scatter_plots : <int>
        How many subplots within the scatter plots to show on one row.

    Returns
    -------
    correlations : <pd.DataFrame>
        The correlation matrix
    fig, ax :
        Figure and axes objects
    '''

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if isinstance(X,np.ndarray):
        X = pd.DataFrame(X,copy=True)

    correlations = X.corr()
    variable_list = list(correlations.columns)
    nr = len(variable_list)

    cax = ax.matshow(correlations, vmin=-1, vmax=1, cmap=color_map)
    fig.colorbar(cax)
    ticks = np.arange(0,correlations.shape[1],1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if variable_names is None:# and isinstance(X,pd.DataFrame):
        variable_names = variable_list

    if variable_names is not None:
        if len(variable_names) != len(variable_list):
            print('*****  Warning: feature_names may not be valid!  *****')

        ax.set_xticklabels(variable_names)
        ax.set_yticklabels(variable_names)
        plt.xticks(rotation=90)

    if scatter_plots == True:
        iu = np.triu_indices(nr)  # indices of upper-triangle elements
        corr_abs = np.abs(np.array(correlations))
        corr_abs[iu] = 0  # only keep lower-triangle elements
        hi_corr_idx = np.where(corr_abs >= thres)  # row & col indices of eligible variables
        indices = list(zip(hi_corr_idx[0],hi_corr_idx[1]))  # variable pairs

        if indices:  # not empty list
            n_cols = min(ncols_scatter_plots, len(indices))
            n_rows = int(np.ceil(len(indices) / float(n_cols)))

            sz = 2.5   # (approx.) size of each subplot (unit: inch)
            mg = 0.28  # (approx.) margin between subplots (unit: inch)
            fig_size_ = (n_cols*sz + mg*(n_cols-1), n_rows*sz + mg*(n_rows-1))
            fig = plt.figure(figsize=fig_size_)
            ax  = plt.axes()

            for j, index in enumerate(indices):
                sub_ax = plt.subplot(n_rows,n_cols,j+1)
                scatter_plot_two_cols(X,[variable_names[i] for i in index],fig,sub_ax)

            plt.tight_layout(h_pad=0.3)
        else:
            print('*****  No two variables have correlation higher than threshold.  *****')

    return fig, ax, correlations

#%%============================================================================
def scatter_plot_two_cols(X, two_columns, fig=None, ax=None,
                          figsize=(3,3), dpi=100, alpha=0.5, color=None,
                          grid_on=True, logx=False, logy=False):
    '''
    Produce scatter plots of two of the columns in X (the data matrix).
    The correlation between the two columns are shown on top of the plot.

    Input
    -----
    X : <pd.DataFrame>
        The dataset. Currently only supports pandas dataframe.
    two_columns : list of two <str> or two <int>
        The names or indices of the two columns within X. Must be a list of
        length 2. The elements must either be both integers, or both strings.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graphs are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : list of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    alpha : scalar
        Opacity of the scatter points.
    color : <str> or list of tuple
        Color of the scatter points. If None, default matplotlib color palette
        will be used.
    grid_on : <bool>
        Whether or not to show grids.

    Returns
    -------
    fig, ax :
        Figure and axes objects
    '''

    from scipy import stats

    fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

    if not isinstance(two_columns,list):
        raise TypeError('"two_columns" must be a list of length 2.')
    if len(two_columns) != 2:
        raise TypeError('Length of "two_columns" must be 2.')

    if isinstance(two_columns[0],str):
        x = X[two_columns[0]]
        xlabel = two_columns[0]
    elif isinstance(two_columns[0],(int,np.integer)):
        x = X.iloc[:,two_columns[0]]
        xlabel = X.columns[two_columns[0]]
    else:
        sys.exit('*****  Error: two_columns must be str list or int list!  *****')

    if isinstance(two_columns[1],str):
        y = X[two_columns[1]]
        ylabel = two_columns[1]
    elif isinstance(two_columns[1],(int,np.integer)):
        y = X.iloc[:,two_columns[1]]
        ylabel = X.columns[two_columns[1]]
    else:
        sys.exit('*****  Error: two_columns must be str list or int list!  *****')

    x = np.array(x)  # convert to numpy array so that x[ind] runs correctly
    y = np.array(y)

    nan_index_in_x = np.where(np.isnan(x))[0]
    nan_index_in_y = np.where(np.isnan(y))[0]
    nan_index = set(nan_index_in_x) | set(nan_index_in_y)
    not_nan_index = list(set(range(len(x))) - nan_index)
    _, _, r_value, _, _ = stats.linregress(x[not_nan_index],y[not_nan_index])

    ax.scatter(x,y,alpha=alpha,color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title('$r$ = %.2f' % r_value)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    if grid_on == True:
        ax.grid(ls=':',lw=0.5)
        ax.set_axisbelow(True)

    return fig, ax

#%%============================================================================
def bin_and_mean(xdata, ydata, bins=10, distribution='normal', show_fig=True,
                 fig=None, ax=None, figsize=None, dpi=100, show_bins=True,
                 raw_data_label='raw', mean_data_label='average', xlabel=None,
                 ylabel=None, logx=False, logy=False, grid_on=True,
                 error_bars_on=False, error_shades_on=True, legend_on=True,
                 subsamp_thres=None):
    '''
    Calculates bin-and-mean results and shows the bin-and-mean plot (optional).

    A "bin-and-mean" plot is a more salient way to show the dependency of ydata
    on xdata. The data points (xdata, ydata) are divided into different groups
    according to the value of x (via the "bins" argument), and within each
    group, the mean values of x and y are calculated, and considered as the
    representative x and y values.

    "bin-and-mean" works better when data points are highly skewed (e.g.,
    a lot of data points for when x is small, but very few for large x). The
    data points when x is large are usually not noises, and could be even more
    valuable (think of the case where x is earthquake magnitude and y is the
    related economic loss). If we want to study the relationship between
    economic loss and earthquake magnitude, we need to bin-and-mean raw data
    and draw conclusions from the mean data points.

    The theory that enables this method is the assumption that the data points
    with similar x values follow the same distribution. Naively, we assume the
    data points are normally distributed, then y_mean is the arithmetic mean of
    the data points within a bin. We also often assume the data points follow
    log-normal distribution (if we want to assert that y values are all
    positive), then y_mean is the expected value of the log-normal distribution,
    while x_mean for any bins are still just the arithmetic mean.

    Notes:
      (1) For log-normal distribution, the expective value of y is:
                    E(Y) = exp(mu + (1/2)*sigma^2)
          and the variance is:
                 Var(Y) = [exp(sigma^2) - 1] * exp(2*mu + sigma^2)
          where mu and sigma are the two parameters of the distribution.
      (2) Knowing E(Y) and Var(Y), mu and sigma can be back-calculated:
                              ___________________
             mu = ln[ E(Y) / V 1 + Var(Y)/E^2(Y)  ]
                      _________________________
             sigma = V ln[ 1 + Var(Y)/E^2(Y) ]

          (Reference: https://en.wikipedia.org/wiki/Log-normal_distribution)

    Parameters
    ----------
    xdata, ydata : <array_like>
        Raw x and y data points (with the same length). Can be pandas Series or
        numpy arrays.
    bins : <int> or <array_like>
        Number of bins (an integer), or an array representing the actual bin
        edges. If bin edges, edges are inclusive on the lower bound, e.g.,
        a value 2 shall fall into the bin [2,3), but not the bin [1,2).
        Note that the binning is done according x values.
    distribution : <str>
        Specifies which distribution the y values within a bin follow. Use
        'lognormal' if you want to assert all positive y values. Only supports
        normal and log-normal distributions at this time.
    show_fig : <bool>
        Whether or not to show a bin-and-mean plot
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    show_bins : <bool>
        Whether or not to show the bin edges as vertical lines on the plots
    raw_data_label, mean_data_label : <str>
        Two strings that specify the names of the raw data and the averaged
        data, respectively, such as "raw data" and "averaged data". Useless
        if show_legend is False.
    xlabel, ylabel : <str>
        Label for the x axis of the plot. If None and xdata is a panda Series,
        use xdata's 'name' attribute as xlabel.
    ylabel : <str>
        Similar to xlabel.
    logx, logy : <bool>
        Whether or not to adjust the scales of x and/or y axes to log
    grid_on : <bool>
        Whether or not to show the grids
    error_bars_on : <bool>
        Whether or not to show error bars (of y values) of each bin
    error_shades_on : <bool>
        Whether or not to show error shades (of y values) of each bin; this
        argument overrides error_bars_on
    legend_on : <bool>
        Whether or not to show the legend

    Returns
    -------
    fig, ax :
        Figure and axes objects
    x_mean, y_mean : <np.ndarray>
        Mean values of x and y for each data group (i.e., "bin")
    y_std : <np.ndarray>
        Standard deviation of y for each data group (i.e., "bin")
    '''

    #------------Pre-process "bins"--------------------------------------------
    if isinstance(bins,(int,np.integer)):  # if user specifies number of bins
        if bins <= 0:
            raise ValueError('"bins" must be a positive integer.')
        else:
            nr = bins + 1  # create bins with percentiles in xdata
            x_uni = np.unique(xdata)
            bins = [np.nanpercentile(x_uni,(j+0.)/bins*100) for j in range(nr)]
            if not all(x <= y for x,y in zip(bins,bins[1:])):  # https://stackoverflow.com/a/4983359/8892243
                print('\nWARNING: Resulting "bins" array is not monotonically '
                      'increasing. Please use a smaller "bins" to avoid potential '
                      'issues.\n')
    elif isinstance(bins,(list,np.ndarray)):  # if user specifies array
        nr = len(bins)
    else:
        raise TypeError('"bins" must be either an integer or an array.')

    #-----------Pre-process xlabel and ylabel----------------------------------
    if not xlabel and isinstance(xdata, pd.Series):  # xdata has 'name' attr
        xlabel = xdata.name
    if not ylabel and isinstance(ydata, pd.Series):  # ydata has 'name' attr
        ylabel = ydata.name

    #-----------Group data into bins-------------------------------------------
    inds = np.digitize(xdata, bins)
    x_mean = np.zeros(nr-1)
    y_mean = np.zeros(nr-1)
    y_std  = np.zeros(nr-1)
    x_subs = []  # subsampled x data (for faster scatter plots)
    y_subs = []
    for j in range(nr-1):  # loop over every bin
        x_in_bin = xdata[inds == j+1]
        y_in_bin = ydata[inds == j+1]

        #------------Calculate mean and std------------------------------------
        if len(x_in_bin) == 0:  # no point falls into current bin
            x_mean[j] = np.nan  # this is to prevent numpy from throwing...
            y_mean[j] = np.nan  #...confusing warning messages
            y_std[j]  = np.nan
        else:
            x_mean[j] = np.nanmean(x_in_bin)
            if distribution == 'normal':
                y_mean[j] = np.nanmean(y_in_bin)
                y_std[j] = np.nanstd(y_in_bin)
            elif distribution in ['log-normal','lognormal','logn']:
                s, loc, scale = stats.lognorm.fit(y_in_bin, floc=0)
                estimated_mu = np.log(scale)
                estimated_sigma = s
                y_mean[j] = np.exp(estimated_mu + estimated_sigma**2.0/2.0)
                y_std[j]  = np.sqrt(np.exp(2.*estimated_mu + estimated_sigma**2.) \
                             * (np.exp(estimated_sigma**2.) - 1) )
            else:
                raise ValueError('Invalid "distribution" value.')

        #------------Pick subsets of data, for faster plotting-----------------
        #------------Note that this does not affect mean and std---------------
        if subsamp_thres is not None and show_fig:
            if not isinstance(subsamp_thres, (int, np.integer)) or subsamp_thres <= 0:
                raise TypeError('subsamp_thres must be a positive integer or None.')
            if len(x_in_bin) > subsamp_thres:
                x_subs.extend(np.random.choice(x_in_bin,subsamp_thres,replace=False))
                y_subs.extend(np.random.choice(y_in_bin,subsamp_thres,replace=False))
            else:
                x_subs.extend(x_in_bin)
                y_subs.extend(y_in_bin)

    #-------------Plot data on figure------------------------------------------
    if show_fig:
        fig, ax = process_fig_ax_objects(fig, ax, figsize, dpi)

        if subsamp_thres:
            xdata = x_subs
            ydata = y_subs
        ax.scatter(xdata,ydata,c='gray',alpha=0.3,label=raw_data_label,zorder=1)
        if error_shades_on:
            if error_bars_on:
                print('***** WARNING: Error shades overriding error bars. *****')
            ax.plot(x_mean,y_mean,'-o',c='orange',lw=2,label=mean_data_label,zorder=3)
            ax.fill_between(x_mean,y_mean+y_std,y_mean-y_std,label='$\pm$ std',
                                facecolor='orange',alpha=0.35,zorder=2.5)
        else:  # error_shades_on False
            if error_bars_on:
                mean_data_label = 'avg $\pm$ std'
                ax.errorbar(x_mean,y_mean,yerr=y_std,ls='-',marker='o',c='orange',
                            lw=2,elinewidth=1,capsize=2,label=mean_data_label,
                            zorder=3)
            else:  # no error bars, nor error shades
                ax.plot(x_mean,y_mean,'-o',c='orange',lw=2,label=mean_data_label,zorder=3)

        ax.set_axisbelow(True)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        if grid_on:
            ax.grid(ls=':')
            ax.set_axisbelow(True)
        if show_bins:
            ylims = ax.get_ylim()
            for k, edge in enumerate(bins):
                lab_ = 'bin edges' if k==0 else None  # only label 1st edge
                ec = get_colors(N=1)[0]
                ax.plot([edge]*2,ylims,'--',c=ec,lw=1.0,zorder=2,label=lab_)
        if legend_on:
            ax.legend(loc='best')

        return fig, ax, x_mean, y_mean, y_std
    else:
        return None, None, x_mean, y_mean, y_std

