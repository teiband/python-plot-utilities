# plot_utils.discrete_histogram

**plot_utils.discrete_histogram**(*x, fig=None, ax=None, color=None, alpha=None, rot=0, logy=False, title='', figsize=(5,3), dpi=100*):

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
              
In the figure, N is the number of values where x = x1, x2, x3, or x4.
And x1, x2, x3, x4, etc. are the discrete values within x.

#### [Parameters]
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
        
#### [Returns]
    fix, ax:
        Figure and axes objects
        
#### [Reference]
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html
    http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#bar-plots
