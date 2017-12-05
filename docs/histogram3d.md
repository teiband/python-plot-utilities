# plot_utils.histogram3d

**plot_utils.histogram3d(*X, bins=10, fig=None, ax=None, figsize=(8,4), dpi=100, elev=30, azim=5, alpha=0.6, data_labels=None, plot_legend=True, plot_xlabel=False, color=None, dx_factor=0.6, dy_factor=0.8, ylabel='Data', zlabel='Counts', \*\*legend_kwargs*):

Plot 3D histograms. 3D histograms are best used to compare the distribution
of more than one set of data.

#### [Parameters]
    X :
        Input data. X can be:
           (1) a 2D numpy array, where each row is one data set;
           (2) a 1D numpy array, containing only one set of data;
           (3) a list of lists, e.g., [[1,2,3],[2,3,4,5],[2,4]], where each
               element corresponds to a data set (can have different lengths);
           (4) a list of 1D numpy arrays.
               [Note: Robustness is not guaranteed for X being a list of
                      2D numpy arrays.]
    bins : <int> or <array_like>
        Bin specifications. Can be:
           (1) An integer, which indicates number of bins;
           (2) An array or list, which specifies bin edges.
               [Note: If an integer is used, the widths of bars across data
                      sets may be different. Thus array/list is recommended.]
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    elev, azim : scalars
        Elevation and azimuth (3D projection view points)
    alpha : scalar
        Opacity of bars
    data_labels : list of <str>
        Names of different datasets, e.g., ['Simulation', 'Measurement'].
        If not provided, generic names ['Dataset #1', 'Dataset #2', ...]
        are used. The data_labels are only shown when either plot_legend or
        plot_xlabel is True.
    plot_legend : <bool>
        Whether to show legends or not
    plot_xlabel : <str>
        Whether to show data_labels of each data set on their respective x
        axis position or not
    color : list of lists, or tuple of tuples
        Colors of each distributions. Needs to be at least the same length as
        the number of data series in X. Can be RGB colors, HEX colors, or valid
        color names in Python. If None, get_colors('tab10',N=N) will be queried.
    dx_factor, dy_factor : scalars
        Width factor 3D bars in x and y directions. For example, if dy_factor
        is 0.9, there will be a small gap between bars in y direction.
    ylabel, zlabel : <str>
        Labels of y and z axes

#### [Returns]
    fix, ax:
        Figure and axes objects

#### [Notes on x and y directions]
    x direction: across data sets (i.e., if we have three datasets, the
                 bars will occupy three different x values)
    y direction: within dataset