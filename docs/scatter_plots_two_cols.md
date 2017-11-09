# plot_utils.scatter_plot_two_cols

**plot_utils.scatter_plot_two_cols**(*X, two_columns, fig=None, ax=None, figsize=(3,3), dpi=100, alpha=0.5, color=None, grid_on=True, logx=False, logy=False*):

Produce scatter plots of two of the columns in X (the data matrix). The correlation between the two columns are shown on top of the plot.

#### [Input]
    X:
        The dataset. Currently only supports pandas dataframe.
    two_columns:
        The names or indices of the two columns within X. Must be a list of
        length 2. The elements must either be both integers, or both strings.
    fig, ax:
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize:
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi:
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    alpha:
        Opacity of the scatter points.
    color:
        Color of the scatter points. If None, default matplotlib color palette
        will be used.
    grid_on:
        Whether or not to show grids.

#### [Returns]
    fix, ax:
        Figure and axes objects
