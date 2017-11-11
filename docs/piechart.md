# plot_utils.piechart

**plot_utils.piechart**(*target_array, class_names=None, fig=None, ax=None,
             figsize=(3,3), dpi=100, colors=None, autopct='%1.1f%%',
             fontsize=None, \*\*piechart_kwargs*):

Plot a pie chart demonstrating the fraction of different classes within an array.

#### [Parameters]
    target_array : <array_like>
        An array containing categorical values (could have more than two
        categories). Target value can be numeric or texts.
    class_names : sequence of str
        Names of different classes. The order should correspond to that in the
        target_array. For example, if target_array has 0 and 1 then class_names
        should be ['0', '1']; and if target_array has "pos" and "neg", then
        class_names should be ['neg','pos'] (i.e., alphabetical).
        If None, values of the categories will be used as names.
    fig, ax :
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : <tuple of int/float>
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : <int, float>
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    colors : <list> or None
        A list of colors (can be RGB values, hex strings, or color names) to be
        used for each class. The length can be longer or shorter than the number
        of classes. If longer, only the first few colors are used; if shorter,
        colors are wrapped around.
        If None, automatically use the Pastel2 color map (8 colors total).
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

#### [Returns]
    fig, ax:
        Figure and axes objects
