# plot_utils.plot_timeseries

**plot_utils.plot_timeseries**(*time_series, date_fmt=None, fig=None, ax=None, figsize=(10,3), xlabel='Time', ylabel=None, label=None, color=None, lw=2, ls='-', marker=None, fontsize=12, xgrid_on=True, ygrid_on=True, title=None, dpi=96, month_grid_width=None*):

Plot time_series, where its index indicates dates (e.g., year, month, date).

You can plot multiple time series by supplying a multi-column pandas Dataframe as time_series, but you cannot use custom line specifications (colors, width, and styles) for each time series. It is recommended to use plot_multiple_timeseries() in stead.

#### [Parameters]
    time_series : <pd.Series> or <pd.DataFrame>
        A pandas Series, with index being date; or a pandas DataFrame, with
        index being date, and each column being a different time series.
    date_fmt : <str>
        Date format specifier, e.g., '%Y-%m' or '%d/%m/%y'.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        figure size (width, height) in inches (fig object passed via
        "fig" will over override this parameter)
    dpi : scalar
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    xlabel : <str>
        Label of X axis. Usually "Time" or "Date"
    ylabel : <str>
        Label of Y axis. Usually the meaning of the data
    label : <str>
        Label of data, for plotting legends
    color : <list> or <str>
        Color of line. If None, let Python decide for itself.
    xgrid_on : <bool>
        Whether or not to show vertical grid lines (default: True)
    ygrid_on : <bool>
        Whether or not to show horizontal grid lines (default: True)
    title : <str>
        Figure title (optional)
    zorder : scalar
        Set the zorder for lines. Higher zorder are drawn on top.
    month_grid_width : <scalar>
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

Note that setting keyword arguments such as color or ls ("linestyle") will force all time series to have the same color or ls. So it is recommended to let this function generate distinguishable line specifications (color/linestyle/linewidth combinations) by itself. (Although the more time series, the less the distinguishability. 240 time series or less is recommended.)

#### [Parameters]
    multiple_time_series : <pandas.DataFrame>
        A pandas dataframe, with index being date , andeach column being a
        different time series.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Figure size (width, height) in inches (fig object passed via
        "fig" will over override this parameter)
    dpi : <scalar>
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    ncol_legend : <int>
        Number of columns of the legend
    **kwargs :
        Other keyword arguments to be passed to plot_timeseries(), such as
        color, marker, fontsize, etc. (Check docstring of plot_timeseries()).

#### [Returns]
    fig, ax:
        Figure and axes objects

---------------------------------------------------------
# plot_utils.fill_timeseries

**plot_utils.fill_timeseries**(*time_series, upper_bound, lower_bound, date_fmt=None, fig=None, ax=None, figsize=(10,3), xlabel='Time', ylabel=None, label=None, color=None, lw=3, ls='-', fontsize=12, title=None, dpi=96, xgrid_on=True, ygrid_on=True*):

Plot time_series as a line, where its index indicates a date (e.g., year, month, date).

And then plot the upper bound and lower bound as shaded areas beneath the line.

#### [Parameters]
    time_series : <pd.Series>
        a pandas Series, with index being date
    upper_bound, lower_bound : <pd.Series>
        upper/lower bounds of the time series, must have the same length as
        time_series
    date_fmt : <str>
        Date format specifier, e.g., '%Y-%m' or '%d/%m/%y'.
    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        figure size (width, height) in inches (fig object passed via "fig"
        will over override this parameter)
    dpi : scalar
        Screen resolution (fig object passed via "fig" will over override
        this parameter)
    xlabel : <str>
        Label of X axis. Usually "Time" or "Date"
    ylabel : <str>
        Label of Y axis. Usually the meaning of the data
    label : <str>
        Label of data, for plotting legends
    color : <str> or list or tuple
        Color of line. If None, let Python decide for itself.
    lw : scalar
        line width of the line that represents time_series
    ls : <str>
        line style of the line that represents time_series
    fontsize : scalar
        font size of the texts in the figure
    title : <str>
        Figure title (optional)
    xgrid_on : <bool>
        Whether or not to show vertical grid lines (default: True)
    ygrid_on : <bool>
        Whether or not to show horizontal grid lines (default: True)

#### [Returns]
    fix, ax:
        Figure and axes objects
