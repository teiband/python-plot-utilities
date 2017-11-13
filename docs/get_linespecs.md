# plot_utils.get_linespecs

**plot_utils.get_linespecs**(*color_scheme='tab10', n_linestyle=4, range_linewidth=[1,2,3], priority='color'*):

Returns a list of distinguishable line specifications (color, line style, and line width combinations).

#### [Parameters]

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

#### [Returns]

    A list whose every element is a dictionary that looks like this:
    {'color': '#1f77b4', 'ls': '-', 'lw': 1}. Each element can then be passed
    as keyword arguments to matplotlib.pyplot.plot() or other similar functions.

-----------------------------------------

# plot_utils.linespecs_demo

**plot_utils.linespecs_demo**(*line_specs, horizontal_plot=False*):

Demonstrate all generated line specifications given by get_linespecs()

#### [Parameter]

    line_spec :
        A list of dictionaries that is the returned value of get_linespecs().
    horizontal_plot : bool
        Whether or not to demonstrate the line specifications in a horizontal plot.

#### [Returns]

    fig, ax : <obj>
        Figure and axes objects.

