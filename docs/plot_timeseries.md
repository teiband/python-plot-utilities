# plot_utils.plot_timeseries

**plot_utils.plot_timeseries**(*time_series, fig=None, ax=None, figsize=(10,3), xlabel='Time', ylabel=None, label=None, color=None, lw=2, ls='-', marker=None, fontsize=12, xgrid_on=True, ygrid_on=True, title=None, dpi=96, month_grid_width=None*):

Plot time_series, where its index indicates dates (e.g., year, month, date).

#### [Parameters]
    time_series:
        A pandas Series, with index being date; or a pandas DataFrame, with
        index being date, and each column being a different time series.
    fig, ax:
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize:
        figure size (width, height) in inches (fig object passed via
        "fig" will over override this parameter)
    dpi:
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    xlabel:
        Label of X axis. Usually "Time" or "Date"
    ylabel:
        Label of Y axis. Usually the meaning of the data
    label:
        Label of data, for plotting legends
    color:
        Color of line. If None, let Python decide for itself.
    xgrid_on:
        Whether or not to show vertical grid lines (default: True)
    ygrid_on:
        Whether or not to show horizontal grid lines (default: True)
    title:
        Figure title (optional)
    month_grid_width:
        the on-figure "horizontal width" that each time interval occupies.
        This value determines how X axis labels are displayed (e.g., smaller
        width leads to date labels being displayed with 90 deg rotation).
        Do not change this unless you really know what you are doing.

#### [Returns]
    fix, ax:
        Figure and axes objects

--------------------------------------------------------
# plot_utils.plot_multiple_timeseries

**plot_utils.plot_multiple_timeseries**(*multiple_time_series, show_legend=True, fig=None, ax=None, figsize=(10,3), dpi=100, ncol_legend=3, \*\*kwargs*):

Plot multiple_time_series, where its index indicates dates (e.g., year, month, date).

#### [Parameters]
    multiple_time_series : <pandas.DataFrame>
        A pandas dataframe, with index being date , andeach column being a
        different time series.
    fig, ax : <matplotlib objects>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : <tuple>
        Figure size (width, height) in inches (fig object passed via
        "fig" will over override this parameter)
    dpi : <scalar>
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    ncol_legend : <int>
        Number of columns of the legend
    **kwargs : <dict>
        Other keyword arguments to be passed to plot_timeseries(), such as
        color, marker, fontsize, etc. (Check docstring of plot_timeseries()).

#### [Returns]
    fig, ax:
        Figure and axes objects

---------------------------------------------------------
# plot_utils.fill_timeseries

**plot_utils.fill_timeseries**(*time_series, upper_bound, lower_bound, fig=None, ax=None, figsize=(10,3), xlabel='Time', ylabel=None, label=None, color=None, lw=3, ls='-', fontsize=12, title=None, dpi=96, xgrid_on=True, ygrid_on=True*):

Plot time_series as a line, where its index indicates a date (e.g., year, month, date).

And then plot the upper bound and lower bound as shaded areas beneath the line.

#### [Parameters]
    time_series:
        a pandas Series, with index being date
    upper_bound, lower_bound:
        upper/lower bounds of the time series, must have the same length as
        time_series
    fig, ax:
        Figure and axes objects.
        If provided, the histograms are plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize:
        figure size (width, height) in inches (fig object passed via "fig"
        will over override this parameter)
    xlabel:
        Label of X axis. Usually "Time" or "Date"
    ylabel:
        Label of Y axis. Usually the meaning of the data
    label:
        Label of data, for plotting legends
    color:
        Color of line. If None, let Python decide for itself.
    lw:
        line width of the line that represents time_series
    ls:
        line style of the line that represents time_series
    fontsize:
        font size of the texts in the figure
    xgrid_on:
        Whether or not to show vertical grid lines (default: True)
    ygrid_on:
        Whether or not to show horizontal grid lines (default: True)
    title:
        Figure title (optional)
    dpi:
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    month_grid_width:
        the on-figure "horizontal width" that each time interval occupies.
        This value determines how X axis labels are displayed (e.g., smaller
        width leads to date labels being displayed with 90 deg rotation).
        Do not change this unless you really know what you are doing.

#### [Returns]
    fix, ax:
        Figure and axes objects
