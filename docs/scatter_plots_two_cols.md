# plot_utils.scatter_plot_two_cols

**plot_utils.scatter_plot_two_cols**(*X, two_columns, fig=None, ax=None, figsize=(3,3), dpi=100, alpha=0.5, color=None, grid_on=True, logx=False, logy=False*):

Produce scatter plots of two of the columns in X (the data matrix). The correlation between the two columns are shown on top of the plot.

#### [Input]
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

#### [Returns]
    fix, ax:
        Figure and axes objects
