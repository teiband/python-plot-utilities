# plot_utils.violin_plots

**violin_plot**(*X, fig=None, ax=None, figsize=None, dpi=100, nan_warning=False,
                showmeans=True, showextrema=False, showmedians=False, vert=False,
                data_names=[], rot=45, name_ax_label=None, data_ax_label=None*):
                
Generates violin plots for a each data set within X. (X contains one more set of data points.)

#### [Parameters]
    X : <pd.DataFrame>, <pd.Series>, <np.ndarray>, or <dict>
        The data to be visualized.

        - pd.DataFrame: each column contains a set of data
        - pd.Series: contains only one set of data
        - np.ndarray:
            + 1D numpy array: only one set of data
            + 2D numpy array: each column contains a set of data
            + higher dimensional numpy array: not allowed
        - dict: each key-value pair is one set of data

        Note that the NaN values in the data are implicitly excluded.

    fig, ax : <mpl.figure.Figure>, <mpl.axes._subplots.AxesSubplot>
        Figure and axes objects.
        If provided, the graph is plotted on the provided figure and
        axes. If not, a new figure and new axes are created.
    figsize : tuple of two scalars
        Size (width, height) of figure in inches. If None, it will be determined
        by how many sets of data are there in X (each set takes up 0.5 inches.)
        (fig object passed via "fig" will over override this parameter)
    dpi : <float>
        Screen resolution. (fig object passed via "fig" will over override
        this parameter)
    nan_warning : <bool>
        Whether to show a warning if there are NaN values in the data.
    showmeans, showextrema, showmedians : <bool>
        Whether or not to show the mean value, extrema, or median value on top
        of the violins.
    vert : <bool>
        Whether to show the violins as vertical
    data_names : <list<str>>
        The names of each data set, to be shown as the axis tick label of each
        data set. If [] or None, it will be determined automatically: if X is a
            - np.ndarray: data_names = ['data_0', 'data_1', 'data_2', ...]
            - pd.Series: data_names = X.name
            - pd.DataFrame: data_names = list(X.columns)
            - dict: data_names = list(X.keys())
    rot : <float>
        The rotation (in degrees) of the data_names when shown as the tick
        labels. If vert is False, rot has no effect.
    name_ax_label, data_ax_label : <str>
        The labels of the name axis and the data axis.
        If vert is True, then the name axis is the x axis, otherwise, y axis.

#### [Returns]
    fig, ax :
        Figure and axes objects