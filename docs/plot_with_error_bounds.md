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
    x, y:
        data points to be plotted as a line (should have the same length)
    upper_bound, lower_bound:
        Upper and lower bounds to be plotted as shaded areas. Should have the
        same length as x and y.
    line_color:
        color of the line of y
    shade_color:
        color of the underlying shades
    shade_alpha:
        transparency of the shades
    linewidth:
        width of the line of y
    legend_loc:
        location of the legend, to be passed directly to plt.legend()
    line_label:
        label of the line of y, to be used in the legend
    shade_label:
        label of the shades, to be used in the legend
    fig, ax:
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    logx, logy:
        Whether or not to show x or y axis scales as log
    grid_on:
        whether or not to show the grids

#### [Returns]
    fix, ax:
        Figure and axes objects
