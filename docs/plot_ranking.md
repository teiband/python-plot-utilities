# plot_utils.plot_ranking

**plot_utils.plot_ranking**(*ranking, fig=None, ax=None, figsize='auto', dpi=100, barh=True, top_n=None, score_ax_label=None, name_ax_label=None, invert_name_ax=False, grid_on=True*):

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

#### [Parameters]

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
        If None, show all categories. top_n > 0 means showing the
        highest top_n categories. top_n < 0 means showing the lowest |top_n|
        categories.
    score_ax_label : <str>
        Label of the score axis (e.g., "Age of pet")
    name_ax_label : <str>
        Label of the "category name" axis (e.g., "Pet name")
    invert_name_ax : <bool>
        Whether to invert the "category name" axis. For example, if
        invert_name_ax is False, then higher values are shown on the top
        for barh=True.
    grid_on : <bool>
        Whether or not to show grids on the plot

#### [Returns]

    fig, ax :
        Figure and axes objects
