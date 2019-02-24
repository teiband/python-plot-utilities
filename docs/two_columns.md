        
# plot_utils.bin_and_mean

**plot_utils.bin_and_mean**(*xdata, ydata, bins=10, distribution='normal', show_fig=True, fig=None, ax=None, figsize=None, dpi=100, show_bins=True, raw_data_label='raw', mean_data_label='average', xlabel=None, ylabel=None, logx=False, logy=False, grid_on=True, error_bounds=True, err_bound_type='shade', legend_on=True, subsamp_thres=None*):

Calculates bin-and-mean results and shows the bin-and-mean plot (optional).

A "bin-and-mean" plot is a more salient way to show the dependency of ydata on xdata. The data points (xdata, ydata) are divided into different groups according to the value of x (via the "bins" argument), and within each group, the mean values of x and y are calculated, and considered as the representative x and y values.

"bin-and-mean" works better when data points are highly skewed (e.g., a lot of data points for when x is small, but very few for large x). The data points when x is large are usually not noises, and could be even more valuable (think of the case where x is earthquake magnitude and y is the related economic loss). If we want to study the relationship between economic loss and earthquake magnitude, we need to bin-and-mean raw data and draw conclusions from the mean data points.

The theory that enables this method is the assumption that the data points with similar x values follow the same distribution. Naively, we assume the data points are normally distributed, then y_mean is the arithmetic mean of the data points within a bin. We also often assume the data points follow log-normal distribution (if we want to assert that y values are all positive), then y_mean is the expected value of the log-normal distribution, while x_mean for any bins are still just the arithmetic mean.

    Notes:
      (1) For log-normal distribution, the expective value of y is:
                    E(Y) = exp(mu + (1/2)*sigma^2)
          and the variance is:
                 Var(Y) = [exp(sigma^2) - 1] * exp(2*mu + sigma^2)
          where mu and sigma are the two parameters of the distribution.
      (2) Knowing E(Y) and Var(Y), mu and sigma can be back-calculated:
                              ___________________
             mu = ln[ E(Y) / V 1 + Var(Y)/E^2(Y)  ]
                      _________________________
             sigma = V ln[ 1 + Var(Y)/E^2(Y) ]

          (Reference: https://en.wikipedia.org/wiki/Log-normal_distribution)

#### [Parameters]
    xdata, ydata : <array_like>
        Raw x and y data points (with the same length). Can be lists, pandas
        Series or numpy arrays.
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
        Whether or not to adjust the scales of x and/or y axes to logarithmic
    grid_on : <bool>
        Whether or not to show the grids
    error_bounds : <bool>
        Whether or not to show error bounds of each bin
    err_bound_type : ['shade', 'bar']
        Type of error bound: shaded area or error bars. It has no effects if
        error_bounds is set to False.
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

-----------------------------------------        

# plot_utils.category_means

**plot_utils.category_means**(_categorical_array, continuous_array, fig=None, ax=None, figsize=None, dpi=100, title=None, xlabel=None, ylabel=None, rot=0, dropna=False, show_stats=True, sort_by='name', vert=True, **violinplot_kwargs_):

Summarize the mean values of entries of y corresponding to each distinct
category in x, and show a violin plot to visualize it. The violin plot will
show the distribution of y values corresponding to each category in x.

Also, a one-way ANOVA test (H0: different categories in x yield same
average y values) is performed, and F statistics and p-value are returned.

#### [Parameters]
    categorical_array : <array_like>
        An vector of categorical values.
    continuous_array : <array_like>
        The target variable whose values correspond to the values in x. Must
        have the same length as x. It is natural that y contains continuous
        values, but if y contains categorical values (expressed as integers,
        not strings), this function should also work.
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
    title : <str>
        The title of the violin plot, usually the name of vector x.
    xlabel : <str>
        The label for the x axis (i.e., categories) of the violin plot. If None
        and x is a pandas Series, use x's 'name' attribute as xlabel.
    ylabel : <str>
        The label for the y axis (i.e., average y values) of the violin plot.
        If None and y is a pandas Series, use y's 'name' attribute as ylabel.
    rot : <float>
        The rotation (in degrees) of the x axis labels
    dropna : <bool>
        Whether or not to exclude N/A records in the data
    show_stats : <bool>
        Whether or not to show the statistical test results (F statistics
        and p-value) on the figure.
    sort_by : <str>
        Option to arrange the different categories in `categorical_array` in
        the violin plot. Valid options are: {'name', 'mean', 'median', None}.
        None means no sorting, i.e., using the hashed order of the category
        names; 'mean' and 'median' mean sorting the violins according to the
        mean/median values of each category; 'name' means sorting the violins
        according to the category names.
    vert : <bool>
        Whether to show the violins as vertical
    **violinplot_kwargs :
        Keyword arguments to be passed to plt.violinplot().
        (https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.violinplot.html)
        Note that this subroutine overrides the default behavior of violinplot:
        showmeans is overriden to True and showextrema to False.

#### [Returns]
    fig, ax :
        Figure and axes objects
    mean_values : <dict>
        A dictionary whose keys are the categories in x, and their corresponding
        values are the mean values in y.
    F_test_result : <tuple>
        A tuple in the order of (F_stat, p_value), where F_stat is the computed
        F-value of the one-way ANOVA test, and p_value is the associated
        p-value from the F-distribution.

-----------------------------------------

# plot_utils.positive_rate

**plot_utils.positive_rate**(*categorical_array, two_classes_array, fig=None, ax=None, figsize=None, dpi=100, barh=True, top_n=None, dropna=False, xlabel=None, ylabel=None, show_stats=True*):

Calculate the proportions of the different categories in vector x that fall
into class "1" (or "True") in vector y, and optionally show a figure.

Also, a Pearson's chi-squared test is performed to test the independence
between x and y. The chi-squared statistics, p-value, and degree-of-freedom
are returned.
    
#### [Parameters]
    categorical_array : <array_like>
        An array of categorical values
    two_class_array : <array_like>
        The target variable containing two classes. Each value in y correspond
        to a value in x (at the same index). Must have the same length as x.
        The second unique value in y will be considered as the positive class
        (for example, "True" in [True, False, True], or "3" in [1,1,3,3,1]).
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars, or None
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If 'auto', the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    barh : <bool>
        Whether or not to show the bars as horizontal (otherwise, vertical).
    top_n : <int>
        Only shows top_n categories (ranked by their positive rate) in the
        figure. Useful when there are too many categories. If None, show all
        categories.
    dropna : <bool>
        If True, ignore entries (in both arrays) where there are missing values
        in at least one array. If False, the missing values are treated as a
        new category, "N/A".
    xlabel, ylabel : <str>
        Axes labels.
    show_stats : <bool>
        Whether or not to show the statistical test results (chi2 statistics
        and p-value) on the figure.
        
#### [Returns]
    fig, ax :
        Figure and axes objects
    pos_rate : <pd.Series>
        The positive rate of each categories in x
    chi2_results : <tuple>
        A tuple in the order of (chi2, p_value, degree_of_freedom)

-----------------------------------------

# plot_utils.contingency_table

**plot_utils.contingency_table**(*array_horizontal, array_vertical, fig=None, ax=None, figsize='auto', dpi=100, color_map='auto', xlabel=None, ylabel=None, dropna=False, rot=45, normalize=True, symm_cbar=True, show_stats=True*):

Calculate and visualize the contingency table from two categorical arrays.
Also perform a Pearson's chi-squared test to evaluate whether the two arrays
are independent.

#### [Parameters]
    array_horizontal : <array_like>
        Array to show as the horizontal margin in the contigency table (i.e.,
        its categories are the column headers)
    array_vertical : <array_like>
        Array to show as the vertical margin in the contigency table (i.e.,
        its categories are the row names)
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars, or 'auto'
        Size (width, height) of figure in inches. (fig object passed via "fig"
        will over override this parameter). If 'auto', the figure size will be
        automatically determined from the number of distinct categories in x.
    dpi : scalar
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    color_map : <str> or <matplotlib.colors.Colormap>
        The color scheme specifications. Valid names are listed in
        https://matplotlib.org/users/colormaps.html.
        If relative_color is True, use diverging color maps (e.g., PiYG, PRGn,
        BrBG, PuOr, RdGy, RdBu, RdYlBu, RdYlGn, Spectral, coolwarm, bwr,
        seismic). Otherwise, use sequential color maps (e.g., viridis, jet).
    xlabel : <str>
        The label for the horizontal axis. If None and x is a pandas Series,
        use x's 'name' attribute as xlabel.
    ylabel : <str>
        The label for the vertical axis. If None and y is a pandas Series, use
        y's 'name' attribute as ylabel.
    dropna : <bool>
        If True, ignore entries (in both arrays) where there are missing values
        in at least one array. If False, the missing values are treated as a
        new category, "N/A".
    rot : <float> or 'vertical' or 'horizontal'
        The rotation of the x axis labels.
    normalize : <bool>
        If True, plot the contingency table as the relative difference between
        the observed and the expected (i.e., (obs. - exp.)/exp. ). If False,
        plot the original "observed frequency".
    symm_cbar : <bool>
        If True, the limits of the color bar are symmetric. Otherwise, the
        limits are the natural minimum/maximum of the table to be plotted.
        It has no effect if "normalize" is set to False.
    show_stats : <bool>
        Whether or not to show the statistical test results (chi2 statistics
        and p-value) on the figure.

#### [Returns]
    fig, ax :
        Figure and axes objects
    chi2_results : <tuple>
        A tuple in the order of (chi2, p_value, degree_of_freedom)
    correlation_metrics : <tuple>
        A tuple in the order of (phi coef., coeff. of contingency, Cramer's V)

