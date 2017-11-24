# plot_utils.plot_with_error_bounds

**plot_utils.plot_with_error_bounds**(*x, y, upper_bound, lower_bound, line_color=[0.4]\*3, shade_color=[0.7]\*3, shade_alpha=0.5, linewidth=2.0, legend_loc='best', line_label='Data', shade_label='`$\mathregular{\pm}$STD`', fig=None, ax=None, logx=False, logy=False, grid_on=True*):

Plot a graph with one line and its upper and lower bounds, with areas between bounds shaded. The effect is similar to this illustration below.

```
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
```

#### [Parameters]
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

#### [Returns]
    fix, ax:
        Figure and axes objects
