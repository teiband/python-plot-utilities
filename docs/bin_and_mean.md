# plot_utils.bin_and_mean

**plot_utils.bin_and_mean**(*xdata, ydata, bins=10, distribution='normal', show_fig=True, fig=None, ax=None, figsize=None, dpi=100, show_bins=True, raw_data_label='raw', mean_data_label='average', xlabel=None, ylabel=None, logx=False, logy=False, grid_on=True, error_bars_on=False, error_shades_on=True, legend_on=True, subsamp_thres=None*):

Calculates bin-and-mean results and shows the bin-and-mean plot (optional).

A "bin-and-mean" plot is a more salient way to show the dependency of ydata on xdata. The data points (xdata, ydata) are divided into different groups according to the value of x (via the "bins" argument), and within each group, the mean values of x and y are calculated, and considered as the representative x and y values.

"bin-and-mean" works better when data points are highly skewed (e.g., a lot of data points for when x is small, but very few for large x). The data points when x is large are usually not noises, and could be even more valuable (think of the case where x is earthquake magnitude and y is the related economic loss). If we want to study the relationship between economic loss and earthquake magnitude, we need to bin-and-mean raw data and draw conclusions from the mean data points.

The theory that enables this method is the assumption that the data points with similar x values follow the same distribution. Naively, we assume the data points are normally distributed, then y_mean is the arithmetic mean of the data points within a bin. We also often assume the data points follow log-normal distribution (if we want to assert that y values are all positive), then y_mean is the expected value of the log-normal distribution, while x_mean for any bins are still just the arithmetic mean.

    Note: For log-normal, the expective value of y is:
                    E(Y) = exp(mu + (1/2)*sigma^2)
          where mu and sigma are the two parameters of the distribution.

#### [Parameters]
    xdata, ydata : <array_like>
        Raw x and y data points (with the same length). Can be pandas Series or
        numpy arrays.
    bins : <int> or <array_like>
        Number of bins (an integer), or an array representing the actual bin
        edges. If bin edges, edges are inclusive on the lower bound, e.g.,
        a value 2 shall fall into the bin [2,3), but not the bin [1,2).
        Note that the binning is done according x values.
    distribution : <str>
        Specifies which distribution the y values within a bin follow. Use
        'lognormal' if you want to assert all positive y values. Only supports
        normal and log-normal distributions at this time.
    show_fig : <bool>
        Whether or not to show a bin-and-mean plot
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
    show_bins : <bool>
        Whether or not to show the bin edges as vertical lines on the plots
    raw_data_label, mean_data_label : <str>
        Two strings that specify the names of the raw data and the averaged
        data, respectively, such as "raw data" and "averaged data". Useless
        if show_legend is False.
    xlabel, ylabel : <str>
        Label for the x axis of the plot. If None and xdata is a panda Series,
        use xdata's 'name' attribute as xlabel.
    ylabel : <str>
        Similar to xlabel.
    logx, logy : <bool>
        Whether or not to adjust the scales of x and/or y axes to log
    grid_on : <bool>
        Whether or not to show the grids
    error_bars_on : <bool>
        Whether or not to show error bars (of y values) of each bin
    error_shades_on : <bool>
        Whether or not to show error shades (of y values) of each bin; this
        argument overrides error_bars_on
    legend_on : <bool>
        Whether or not to show the legend
    subsamp_thres : <int>
        A positive integer that defines the number of data points in each bin
        to show in the scatter plot. The smaller this number, the faster the
        plotting process. If larger than the number of data points in a bin,
        then all data points from that bin are plotted. If None, then all data
        points from all bins are plotted.

#### [Returns]
    fig, ax :
        Figure and axes objects
    x_mean, y_mean : <np.ndarray>
        Mean values of x and y for each data group (i.e., "bin")
    y_std : <np.ndarray>
        Standard deviation of y for each data group (i.e., "bin")
