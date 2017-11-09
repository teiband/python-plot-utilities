# plot_utils.histogram3d

**plot_utils.histogram3d**(*X, bins=10, fig=None, ax=None, elev=30, azim=5, alpha=0.6, data_labels=None, plot_legend=True, plot_xlabel=False, dx_factor=0.6, dy_factor=0.8, ylabel='Data', zlabel='Counts', \*\*legend_kwargs*):


Plot 3D histograms. 3D histograms are best used to compare the distribution
of more than one set of data.

#### [Parameters]
    X: 
        Input data. X can be:
           (1) a 2D numpy array, where each row is one data set;
           (2) a 1D numpy array, containing only one set of data;
           (3) a list of lists, e.g., [[1,2,3],[2,3,4,5],[2,4]], where each
               element corresponds to a data set (can have different lengths);
           (4) a list of 1D numpy arrays.
               [Note: Robustness is not guaranteed for X being a list of
                      2D numpy arrays.]
    bins: 
        Bin specifications. Can be:
           (1) An integer, which indicates number of bins;
           (2) An array or list, which specifies bin edges.
               [Note: If an integer is used, the widths of bars across data
                      sets may be different. Thus array/list is recommended.]
    fig, ax:
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    elev, azim:
        Elevation and azimuth (3D projection view points)
    alpha:
        Opacity of bars
    data_labels:
        Names of different datasets, e.g., ['Simulation', 'Measurement'].
        If not provided, generic names ['Dataset #1', 'Dataset #2', ...]
        are used. The data_labels are only shown when either plot_legend or
        plot_xlabel is True.
    plot_legend:
        Whether to show legends or not
    plot_xlabel:
        Whether to show data_labels of each data set on their respective x
        axis position or not
    dx_factor, dy_factor:
        Width factor 3D bars in x and y directions. For example, if dy_factor
        is 0.9, there will be a small gap between bars in y direction.
    ylabel, zlabel: 
        Labels of y and z axes

#### [Returns]
    fix, ax:
        Figure and axes objects

#### [Notes on x and y directions]
    x direction: across data sets (i.e., if we have three datasets, the
                 bars will occupy three different x values)
    y direction: within dataset