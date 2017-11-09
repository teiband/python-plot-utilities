# plot_utils.plot_correlation

**plot_utils.plot_correlation**(*X, color_map='RdBu_r', fig=None, ax=None, figsize=(6,6), dpi=100, variable_names=None, scatter_plots=False, thres=0.7, ncols_scatter_plots=3*):

Plot correlation matrix of a dataset X, whose columns are different variables (or a sample of a certain random variable).

#### [Parameters]
    X:
        The data set. Can be a numpy array or pandas dataframe.
    color_map:
        The color scheme to show high, low, negative high correlations. Legit
        names are listed in https://matplotlib.org/users/colormaps.html. Using
        diverging color maps are recommended: PiYG, PRGn, BrBG, PuOr, RdGy,
        RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr, seismic
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
    variable_names:
        Names of the variables in X. If X is a pandas dataframe, then this
        argument is not need: column names of X is used as variable names. If
        X is a numpy array, and this argument is not provided, then column
        indices are used. The length of variable_names should match the number
        of columns in X; if not, a warning will be thrown (but not error).
    scatter_plots:
        Whether or not to show the variable pairs with high correlation.
        Variable pairs whose absolute value of correlation is higher than thres
        will be plotted as scatter plots.
    thres:
        Threshold of correlation (absolute value). Variable pairs whose absolute
        correlation is higher than thres will be plotted as scatter plots.
    ncols_scatter_plots:
        How many subplots within the scatter plots to show on one row.

#### [Returns]
    correlations:
        The correlation matrix
    fix, ax:
        Figure and axes objects

