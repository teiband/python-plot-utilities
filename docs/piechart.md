# plot_utils.piechart

**plot_utils.piechart**(*target_array, class_names=None, dropna=False, top_n=None, sort_by='counts', fig=None, ax=None, figsize=(3,3), dpi=100, colors=None, display='percent', title=None, fontsize=None, verbose=True, \*\*piechart_kwargs*):

Plot a pie chart demonstrating proportions of different categories within an array.

#### [Parameters]
    target_array : <array_like>
        An array containing categorical values (could have more than two
        categories). Target value can be numeric or texts.
    class_names : sequence of str
        Names of different classes. The order should correspond to that in the
        target_array. For example, if target_array has 0 and 1 then class_names
        should be ['0', '1']; and if target_array has "pos" and "neg", then
        class_names should be ['neg','pos'] (i.e., alphabetical).
        If None, values of the categories will be used as names. If [], then
        no class names are displayed.
    dropna : <bool>
        Whether to drop nan values or not. If False, they show up as 'N/A'.
    top_n : <int>
        An integer between 1 and the number of unique categories in target_array.
        Useful for preventing plotting too many unique categories (very slow).
        If None, plot all categories.
    sort_by : {'counts', 'name'}
        An option to control whether the pie slices are arranged by the counts
        of each unique categories, or by the names of those categories.
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
    title : <str> or None
        The text to be shown on the top of the pie chart
    fontsize : scalar or tuple/list of two scalars
        Font size. If scalar, both the class names and the percentages are set
        to the specified size. If tuple of two scalars, the first value sets
        the font size of class names, and the last value sets the font size
        of the percentages.
    verbose : <bool>
        Whether or to show a "Plotting more than 100 slices; please be patient"
        message when the number of categories exceeds 100.
    **piechart_kwargs :
        Keyword arguments to be passed to matplotlib.pyplot.pie function,
        except for "colors", "labels"and  "autopct" because this subroutine
        re-defines these three arguments.
        (See https://matplotlib.org/api/_as_gen/matplotlib.pyplot.pie.html)

#### [Returns]
    fig, ax:
        Figure and axes objects
