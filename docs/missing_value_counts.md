# plot_utils.missing_value_counts

**missing_value_counts**(*X, fig=None, ax=None, figsize=(12,3), dpi=100, rot=90*):

Visualize the number of missing values in each column of X.

#### [Parameters]
    X : <pd.DataFrame> or <pd.Series>
        Input data set whose every row is an observation and every column is
        a variable.
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
    rot : <float>
        Rotation (in degrees) of the x axis labels

#### [Returns]
    fig, ax :
        Figure and axes objects
    null_counts : <pd.Series>
        A pandas Series whose every element is the number of missing values
        corresponding to each column of X.
