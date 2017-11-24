# plot_utils.plot_correlation

**plot_utils.plot_correlation**(*X, color_map='RdBu_r', fig=None, ax=None, figsize=(6,6), dpi=100, variable_names=None, scatter_plots=False, thres=0.7, ncols_scatter_plots=3*):

Plot correlation matrix of a dataset X, whose columns are different variables (or a sample of a certain random variable).

#### [Parameters]
    X : <np.ndarray> or <pd.DataFrame>
        The data set.
    color_map : <str> or <matplotlib.colors.Colormap>
        The color scheme to show high, low, negative high correlations. Legit
        names are listed in https://matplotlib.org/users/colormaps.html. Using
        diverging color maps are recommended: PiYG, PRGn, BrBG, PuOr, RdGy,
        RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr, seismic
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    variable_names : list of <str>
        Names of the variables in X. If X is a pandas dataframe, then this
        argument is not need: column names of X is used as variable names. If
        X is a numpy array, and this argument is not provided, then column
        indices are used. The length of variable_names should match the number
        of columns in X; if not, a warning will be thrown (but not error).
    scatter_plots : bool
        Whether or not to show the variable pairs with high correlation.
        Variable pairs whose absolute value of correlation is higher than thres
        will be plotted as scatter plots.
    thres : scalar between 0 and 1
        Threshold of correlation (absolute value). Variable pairs whose absolute
        correlation is higher than thres will be plotted as scatter plots.
    ncols_scatter_plots : <int>
        How many subplots within the scatter plots to show on one row.

#### [Returns]
    correlations : <pd.DataFrame>
        The correlation matrix
    fig, ax :
        Figure and axes objects

