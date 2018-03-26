# plot_utils.discrete_histogram

**plot_utils.discrete_histogram**(*x, fig=None, ax=None, figsize=(5,3), dpi=100, color=None, alpha=None, rot=0, logy=False, title=None, xlabel=None, ylabel='Number of occurrences', show_xticklabel=True*):

Plot a discrete histogram based on "x", such as below:

```
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
```
              
In the figure, N is the number of occurences for x1, x2, x3, x4, etc. And x1, x2, x3, x4, etc. are the sorted discrete values within x.

#### [Parameters]
    x : <array_like> or <dict>
        Data to be visualized.
        If x is an array (list, numpy arrary), the content of x is analyzed and
        counts of x's values are plotted.
        If x is a Python dict, then x's keys are treated as discrete values and
        x's values are treated as counts. (plot_ranking() does similar tasks.)
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars, or 'auto'
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If 'auto', the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : <float> or <int>
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    color : <str> or <list>
        Color of bar. If not specified, the default color (muted blue)
        is used.
    alpha : <float>
        Opacity of bar. If not specified, the default value (1.0) is used.
    rot : <float> or <int>
        Rotation angle (degrees) of x axis label. Default = 0 (upright label)
    logy : <bool>
        Whether or not to use log scale for y
    title, xlabel, ylabel : <str>
        The title, x label, and y label
    show_xticklabel : <bool>
        Whether or not to show the x tick labels (the names of the classes)
        
#### [Returns]
    fix, ax:
        Figure and axes objects
        
#### [Reference]
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html
    http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#bar-plots
