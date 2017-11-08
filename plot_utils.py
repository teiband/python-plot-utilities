# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:37:26 2017

@author: Jian
"""

import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib as mpl
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#%%############################################################################
def histogram3d(X,bins=10,fig=None,ax=None,
                elev=30,azim=5,alpha=0.6,data_labels=None,
                plot_legend=True,plot_xlabel=False,
                dx_factor=0.6,dy_factor=0.8,
                ylabel='Data',zlabel='Counts',
                **legend_kwargs):
    '''
    Plot 3D histograms. 3D histograms are best used to compare the distribution
    of more than one set of data.

    [Notes on x and y directions]
        x direction: across data sets (i.e., if we have three datasets, the
                     bars will occupy three different x values)
        y direction: within dataset

    [Inputs]
        X: Input data. X can be:
               (1) a 2D numpy array, where each row is one data set;
               (2) a 1D numpy array, containing only one set of data;
               (3) a list of lists, e.g., [[1,2,3],[2,3,4,5],[2,4]], where each
                   element corresponds to a data set (can have different lengths);
               (4) a list of 1D numpy arrays.
                   [Note: Robustness is not guaranteed for X being a list of
                          2D numpy arrays.]
        bins: Bin specifications. Can be:
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
        ylabel, zlabel: Labels of y and z axes

    [Outputs]
      - fig: figure object for the user to manipulate after calling this function
      - ax: axes object for the user to manipulate after calling this function
    '''

    from mpl_toolkits.mplot3d import Axes3D

    if type(X) is np.ndarray:
        if X.ndim <= 1:  # X is a 1D numpy array
            N = 1
            X = [list(X)]  # e.g., np.array([1,2,3]) --> [[1,2,3]], so that X[0] = [1,2,3]
        elif X.ndim == 2:  # X is a 2D numpy array
            N = X.shape[0]  # number of separate distribution to be compared
            X = list(X)  # then turn X into a list of numpy arrays (no longer a 2D numpy array)
        else:  # 3D numpy array or above
            print('*****  If X is a numpy array, it should be a 1D or 2D array.  *****')
            sys.exit()
    elif len(list(X)) > 1:  # adding list() to X to make sure len() does not throw an error
        N = len(X)  # number of separate distribution to be compared
    else:  # X is a scalar
        print('*****  X should either be a list or a 2D numpy array.  *****')
        sys.exit()

    if data_labels is None:
        data_labels = [[None]] * N
        for j in range(N):
            data_labels[j] = 'Dataset #%d' % (j+1)  # use generic data set names

    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=(8,4),dpi=96,facecolor='w',edgecolor='k')
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes(projection='3d')  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    ax.view_init(elev,azim)  # set view elevation and angle

    proxy = [[None]] * N  # create a 'proxy' to help generating legends
    c = get_colors(N,10)  # get a list of colors
    xpos_list = [[None]] * N  # pre-allocation
    for j in range(N):  # loop through each dataset
        if type(bins) == list and len(bins) > 1:  # if 'bins' is a list and length > 1
            bar_width = np.min(np.array(bins[1:])-np.array(bins[:-1]))  # pick the mininum bin width as bar_width
        elif type(bins) == int:  # if 'bins' is an integer (i.e., number of bins)
            bar_width = (np.max(X[j])-np.min(X[j]))/float(bins)  # use the most narrow bin width as bar_width
        else:  # for other type of "bins", try to convert it into a list
            bins = list(bins)
            bar_width = np.min(np.array(bins[1:])-np.array(bins[:-1]))  # pick the mininum bin width as bar_width

        dz, ypos_ = np.histogram(X[j],bins)  # calculate counts and bin edges
        ypos = np.mean(np.array([ypos_[:-1],ypos_[1:]]),axis=0)  # mid-point of all bins
        xpos = np.ones_like(ypos) * (j-0.5)  # location of each data set
        zpos = np.zeros_like(xpos)  # should be all 0's
        dx = bar_width * dx_factor  # width of bars in x direction (across data sets)
        dy = bar_width * dy_factor  # width of bars in y direction (within data set)
        if float(mpl.__version__.split('.')[0]) >= 2.0:
            bar3d_kwargs = {'alpha':alpha}  # 'lw' argument clashes with alpha in 2.0+ versions
        else:
            bar3d_kwargs = {'alpha':alpha, 'lw':0.5}
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=c[j], **bar3d_kwargs)
        proxy[j] = plt.Rectangle((0, 0), 1, 1, fc=c[j])  # generate proxy for plotting legends
        xpos_list[j] = xpos[0] + dx/2.0  # '+dx/2.0' makes x ticks pass through center of bars

    if plot_legend is True:
        default_kwargs = {'loc':9, 'fancybox':True, 'framealpha':0.5,
                          'ncol':N, 'fontsize':10}
        if legend_kwargs == {}:  # set default legend keyword arguments
            legend_kwargs.update(default_kwargs)  # update with default_kwargs
        else:  # if user provides (some of the) keyword arguments
            default_kwargs.update(legend_kwargs)  # use provided args to update default args
            legend_kwargs = default_kwargs  # then replace legend_kwargs with default_kwargs
        ax.legend(proxy,data_labels,**legend_kwargs)

    if plot_xlabel is True:
        ax.set_xticks(xpos_list)  # show x ticks
        ax.set_xticklabels(data_labels)  # use data_labels to denote X ticks
    else:
        ax.set_xticks([None])  # do not show X ticks
        ax.set_xticklabels([] * len(data_labels))  # do not show X tick labels

    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.invert_xaxis()  # make X[0] appear in front, and X[-1] appear at back

    plt.tight_layout(pad=0.3)

    return fig, ax

#%%############################################################################
def get_colors(N,color_scheme='tab10'):
    '''
    Returns a list of N distinguisable colors. When N is larger than the color
    scheme capacity, the color cycle is wrapped around.
    Use color_scheme to specify which color scheme is desired.

    References:
        https://matplotlib.org/users/dflt_style_changes.html#colors-color-cycles-and-color-maps
        https://github.com/vega/vega/wiki/Scales#scale-range-literals
    '''

    if color_scheme == 0:  # default matplotlib color scheme
        palette = ['b','g','r','c','m','y','k']
    elif (color_scheme == 10) or (color_scheme == 'tab10'):
        palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                   '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    elif color_scheme == 'set1':  # nine distinct colors
        palette = [[0.894,0.102,0.110],
                   [0.216,0.494,0.722],
                   [0.302,0.686,0.290],
                   [0.596,0.306,0.639],
                   [1.000,0.498,0.000],
                   [1.000,1.000,0.200],
                   [0.651,0.337,0.157],
                   [0.969,0.506,0.749],
                   [0.600,0.600,0.600],
                   [0.600,0.600,0.600]]
    elif color_scheme == 8.3:  # MATLAB ver 8.3 (R2014a) and earlier
        palette = [[0, 0, 1.0000],  # blue
                   [0, 0.5000, 0],  # green
                   [1.0000, 0, 0],  # red
                   [0, 0.7500, 0.7500],  # cyan
                   [0.7500, 0, 0.7500],  # magenta
                   [0.7500, 0.7500, 0],  # dark yellow
                   [0.2500, 0.2500, 0.2500]]  # dark gray
    elif color_scheme == 8.4:  # MATLAB ver 8.4 (R2014b) and later
        palette = [[0.0000, 0.4470, 0.7410],
                   [0.8500, 0.3250, 0.0980],
                   [0.9290, 0.6940, 0.1250],
                   [0.4940, 0.1840, 0.5560],
                   [0.4660, 0.6740, 0.1880],
                   [0.3010, 0.7450, 0.9330],
                   [0.6350, 0.0780, 0.1840]]

    L = len(palette)
    return [palette[i % L] for i in range(N)]  # wrap around 'palette' if N > L

#%%############################################################################
def find_axes_lim(data_limit,tick_base_unit,direction='upper'):
    '''
    Return a "whole" number to be used as the upper or lower limit of axes.

    For example, if the maximum x value of the data is 921.5, and you would
    like the upper x_limit to be a multiple of 50, then this function returns
    950.

    [Inputs]
        data_limit: The upper and/or lower limit(s) of data.
                    (1) If a tuple (or list) of two elements is provided, then
                        the upper and lower axis limits are automatically
                        determined. (The order of the two elements does not
                        matter.)
                    (2) If a scalar (float or int)is provided, then the axis
                        limit is determined based on the DIRECTION provided.
        tick_base_unit: For example, if you want your axis limit(s) to be a
                        multiple of 20 (such as 80, 120, 2020, etc.), then use
                        20.
        direction: 'upper' or 'lower'; used only when data_limit is a scalar.
                   If data_limit is a tuple/list, then this variable is
                   disregarded.
    [Returns]
        If data_limit is a list/tuple of length 2, return [min_limit,max_limit]
        (Note: it is always ordered no matter what the order of data_limit is.)

        If data_limit is a scalar, return axis_limit according to the DIRECTION.
    '''
    if isinstance(data_limit,float) or isinstance(data_limit,int):  # is scalar
        if direction == 'upper':
            return tick_base_unit * (int(data_limit/tick_base_unit)+1)
        elif direction == 'lower':
            return tick_base_unit * (int(data_limit/tick_base_unit))
        else:
            print('*****  Length of data_limit should be at least 1.  *****')
    elif isinstance(data_limit,list):
        if len(data_limit) > 2:
            print('*****  Length of data_limit should be at most 2.  *****')
            sys.exit()
        elif len(list(data_limit)) == 2:
            min_data = min(data_limit)
            max_data = max(data_limit)
            max_limit = tick_base_unit * (int(max_data/tick_base_unit)+1)
            min_limit = tick_base_unit * (int(min_data/tick_base_unit))
            return [min_limit, max_limit]
        elif len(data_limit) == 1:  # such as [2.14]
            return find_axes_lim(data_limit[0],tick_base_unit,direction) # recursion
    else: # for example, a numpy array...
        if type(data_limit) == np.ndarray:
            data_limit = data_limit.flatten()  # convert np.array(2.5) into np.array([2.5])
            if data_limit.size == 1:
                return find_axes_lim(data_limit[0],tick_base_unit,direction) # recursion
            elif data_limit.size == 2:
                return find_axes_lim(list(data_limit),tick_base_unit,direction) # recursion
            elif data_limit.size >= 3:
                print('*****  Length of data_limit should be at most 2.  *****')
                sys.exit()
            else:
                print('*****  Unrecognized data type for data_limit.  *****')
                sys.exit()
        else:
            print('*****  Unrecognized data type for data_limit.  *****')
            sys.exit()

#%%############################################################################
def discrete_histogram(x,fig=None,ax=None,color=None,alpha=None,
                       rot=0,logy=False,title='',figsize=(5,3),dpi=100):
    '''
    Plot a discrete histogram based on "x", such as below:


      N ^
        |
        |           ____
        |           |  |   ____
        |           |  |   |  |
        |    ____   |  |   |  |
        |    |  |   |  |   |  |
        |    |  |   |  |   |  |   ____
        |    |  |   |  |   |  |   |  |
        |    |  |   |  |   |  |   |  |
       -|--------------------------------------->  x
              x1     x2     x3     x4    ...

    In the figure, N is the number of values where x = x1, x2, x3, or x4.
    And x1, x2, x3, x4, etc. are the discrete values within x.

    [Inputs]
       x:  A list of numpy array that contain the data to be visualized.
       fig, ax:
            Figure and axes objects.
            If provided, the histograms are plotted on the provided figure and
            axes. If not, a new figure and new axes are created.
       color: Color of bar. If not specified, the default color (muted blue)
               is used.
       alpha: Opacity of bar. If not specified, the default value (1.0) is used.
       rot: Rotation angle (degrees) of x axis label. Default = 0 (upright label)

    [Outputs]
      - fig: figure object for the user to manipulate after calling this function
      - ax: axes object for the user to manipulate after calling this function
    [Reference]
    http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.plot.html
    http://pandas.pydata.org/pandas-docs/version/0.18.1/visualization.html#bar-plots
    '''

    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=figsize,dpi=dpi)
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes()  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    X = pd.Series(x)  # convert x into series
    value_count = X.value_counts().sort_index()  # count distinct values and sort
    ax = value_count.plot.bar(color=color,alpha=alpha,ax=ax,rot=rot)#,logy=logy,ylim=(1,10000))
    ax.set_ylabel('Number of occurrences')
    if np.abs(rot) > 0:
        ax.set_xticklabels(value_count.index,rotation=rot,ha='right')
    if logy:   # http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.yscale
        ax.set_yscale('log', nonposy='clip')  # https://stackoverflow.com/a/17952890
    ax.set_title(title)

    return fig, ax

#%%############################################################################
from matplotlib.ticker import ScalarFormatter
class FixedOrderFormatter(ScalarFormatter):
    '''
    Formats axis ticks using scientific notation with a constant order of
    magnitude.

    (Reference: https://stackoverflow.com/a/3679918)
    '''

    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

#%%############################################################################
def choropleth_map_state(data_per_state,vmin=None,vmax=None,map_title='USA map',
                   unit='',cmap='hot_r',fontsize=14,cmap_midpoint=None,
                   figsize=(10,7),dpi=100,shapefile_dir=None):
    '''
    Generate a choropleth map of USA (including Alaska and Hawaii), on a state
    level.

    According to wikipedia, a choropleth map is a thematic map in which areas
    are shaded or patterned in proportion to the measurement of the statistical
    variable being displayed on the map, such as population density or
    per-capita income.

    [Inputs]
      - data_per_state:
          Numerical data of each state, to be plotted onto the map.
          Acceptable data types include:
            - pandas Series: Index should be valid state identifiers (i.e.,
                             state full name, abbreviation, or FIPS code)
            - pandas DataFrame: The dataframe can have only one columns (with
                                the index being valid state identifiers), two
                                columns (with one of the column named 'state',
                                'State', or 'FIPS_code', and containing state
                                identifiers).
            - dictionary: with keys being valid state identifiers, and values
                          being the numerical values to be visualized
      - vmin: Minimum value to be shown on the map. If vmin is larger than the
              actual minimum value in the data, some of the data values will be
              "clipped". This is useful if there are extreme values in the data
              and you do not want those values to complete skew the color
              distribution.
      - vmax: Maximum value to be shown on the map. Similar to vmin.
      - map_title: Title of the map, to be shown on the top of the map.
      - unit: Unit of the numerical (for example, "population per km^2"), to be
              shown on the right side of the color bar.
      - cmap: Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
              for plotting deviation maps.
      - fontsize: Font size of all the texts on the map.
      - cmap_midpoint:
          A numerical value that specifies the "deviation point". For example,
          if your data ranges from -200 to 1000, and you want negative values
          to appear blue-ish, and positive values to appear red-ish, then you
          can set cmap_midpoint to 0.0.
      - figsize: Size of the whole figure (including map and color bar).
      - dpi: On-screen resolution.
      - shapefile_dir:
          Directory where shape files are stored. Shape files (state level and
          county level) should be organized as follows:
              [shapefile_dir]/usa_states/st99_d00.(...)
              [shapefile_dir]/usa_counties/cb_2016_us_county_500k.(...)

    [Outputs]
      - fig: figure object for the user to manipulate after calling this function
      - ax: axes object for the user to manipulate after calling this function

    [References]
        I based my modifications partly on some code snippets in this
        stackoverflow thread:
            https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
    '''
    from mpl_toolkits.basemap import Basemap as Basemap
    from matplotlib.colors import rgb2hex, Normalize
    from matplotlib.patches import Polygon
    from matplotlib.colorbar import ColorbarBase

    if isinstance(data_per_state,pd.Series):
        data_per_state = data_per_state.to_dict()  # convert to dict
    elif isinstance(data_per_state,pd.DataFrame)  and data_per_state.shape[1] == 1:
        data_per_state = data_per_state.iloc[:,0].to_dict()
    elif isinstance(data_per_state,pd.DataFrame) and data_per_state.shape[1] == 2:
        if 'FIPS_code' in data_per_state.columns:
            data_per_state = data_per_state.set_index('FIPS_code')
        elif 'state' in data_per_state.columns:
            data_per_state = data_per_state.set_index('state')
        elif 'State' in data_per_state.columns:
            data_per_state = data_per_state.set_index('State')
        else:
            sys.exit('------  Input data format not recognized!   ----------')
        data_per_state = data_per_state.iloc[:,0].to_dict()

    #  if dict keys are state abbreviations such as "AK", "CA", etc.
    if len(list(data_per_state.keys())[0])==2 and list(data_per_state.keys())[0].isalpha():
        data_per_state = translate_state_abbrev(data_per_state) # convert from 'AK' to 'Alaska'

    #  if dict keys are state FIPS codes such as "01", "45", etc.
    if len(list(data_per_state.keys())[0])==2 and list(data_per_state.keys())[0].isdigit():
        data_per_state = convert_FIPS_to_state_name(data_per_state) # convert from '01' to 'Alabama'

    data_per_state = check_all_states(data_per_state)  # see function definition of check_all_states()

    fig = plt.figure(figsize=(10,7),dpi=dpi)

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(llcrnrlon=-190,llcrnrlat=20,urcrnrlon=-143,urcrnrlat=46,
                projection='merc',lat_ts=20)  # do not change these numbers

    #---------   draw state boundaries  ----------------------------------------
    if shapefile_dir is None:
        shapefile_dir = './shapefiles'
    shp_path_state = os.path.join(shapefile_dir,'usa_states','st99_d00')
    shp_info = m.readshapefile(shp_path_state,'states',drawbounds=True,
                               linewidth=0.45,color='gray')
    shp_info_ = m_.readshapefile(shp_path_state,'states',drawbounds=False)

    #-------- choose a color for each state based on population density. -------
    colors={}
    statenames=[]
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(list(data_per_state.values()))
    if vmax is None:
        vmax = np.nanmax(list(data_per_state.values()))
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            data_ = data_per_state[statename]
            if not np.isnan(data_):
                # calling colormap with value between 0 and 1 returns
                # rgba value. Take sqrt root to spread out colors more.
                colors[statename] = cmap(np.sqrt((data_-vmin)/(vmax-vmin)))[:3]
            else:  # if data_ is NaN, set color to light grey, and with hatching pattern
                colors[statename] = None #np.nan#[0.93]*3
        statenames.append(statename)

    #---------  cycle through state names, color each one.  --------------------
    ax = plt.gca() # get current axes instance

    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            if colors[statenames[nshape]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=[0.4]*3,hatch='\\')
            else:
                color = rgb2hex(colors[statenames[nshape]])
                poly = Polygon(seg,facecolor=color,edgecolor=color)

            ax.add_patch(poly)

    AREA_1 = 0.005  # exclude small Hawaiian islands that are smaller than AREA_1
    AREA_2 = AREA_1 * 30.0  # exclude Alaskan islands that are smaller than AREA_2
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for nshape, shapedict in enumerate(m_.states_info):  # plot Alaska and Hawaii as map insets
        if shapedict['NAME'] in ['Alaska', 'Hawaii']:
            seg = m_.states[int(shapedict['SHAPENUM'] - 1)]
            if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > AREA_1:
                seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            elif shapedict['NAME'] == 'Alaska' and float(shapedict['AREA']) > AREA_2:
                seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y)\
                       for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor='gray', linewidth=.45)
            ax.add_patch(poly)

    ax.set_title(map_title)

    #---------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8]*3  # define light gray color RGB
    m_.plot(np.linspace(170,177),np.linspace(29,29),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(177,180),np.linspace(29,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(180,180),np.linspace(26,23),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-177),np.linspace(23,20),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-175),np.linspace(26,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-175,-171),np.linspace(26,22),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-171,-171),np.linspace(22,20),linewidth=1.,
            color=light_gray,latlon=True)

    #---------   Show color bar  ---------------------------------------
    if cmap_midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=cmap_midpoint)

    ax_c = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical',
                      label=unit)

    mpl_version = mpl.__version__.split('.')
    if float(mpl_version[0] + '.' + mpl_version[0]) >= 2.1: # version > 2.1.0
        cb = adjust_colorbar_tick_labels(cb,
                                         np.nanmax(list(data_per_state.values())) > vmax,
                                         np.nanmin(list(data_per_state.values())) < vmin)

    #---------   Set overall font size  --------------------------------
    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig,ax  # return figure and axes handles

#%%############################################################################
def choropleth_map_county(data_per_county,vmin=None,vmax=None,unit='',cmap='hot_r',
                          map_title='USA county map',fontsize=14,cmap_midpoint=None,
                          figsize=(10,7),dpi=100,shapefile_dir=None):
    '''
    Generate a choropleth map of USA (including Alaska and Hawaii), on a county
    level.

    According to wikipedia, a choropleth map is a thematic map in which areas
    are shaded or patterned in proportion to the measurement of the statistical
    variable being displayed on the map, such as population density or
    per-capita income.

    [Inputs]
      - data_per_county:
          Numerical data of each county, to be plotted onto the map.
          Acceptable data types include:
            - pandas Series: Index should be valid county identifiers (i.e.,
                             5 digit county FIPS codes)
            - pandas DataFrame: The dataframe can have only one columns (with
                                the index being valid county identifiers), two
                                columns (with one of the column named 'state',
                                'State', or 'FIPS_code', and containing county
                                identifiers).
            - dictionary: with keys being valid county identifiers, and values
                          being the numerical values to be visualized
      - vmin: Minimum value to be shown on the map. If vmin is larger than the
              actual minimum value in the data, some of the data values will be
              "clipped". This is useful if there are extreme values in the data
              and you do not want those values to complete skew the color
              distribution.
      - vmax: Maximum value to be shown on the map. Similar to vmin.
      - map_title: Title of the map, to be shown on the top of the map.
      - unit: Unit of the numerical (for example, "population per km^2"), to be
              shown on the right side of the color bar.
      - cmap: Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
              for plotting deviation maps.
      - fontsize: Font size of all the texts on the map.
      - cmap_midpoint:
          A numerical value that specifies the "deviation point". For example,
          if your data ranges from -200 to 1000, and you want negative values
          to appear blue-ish, and positive values to appear red-ish, then you
          can set cmap_midpoint to 0.0.
      - figsize: Size of the whole figure (including map and color bar).
      - dpi: On-screen resolution.
      - shapefile_dir:
          Directory where shape files are stored. Shape files (state level and
          county level) should be organized as follows:
              [shapefile_dir]/usa_states/st99_d00.(...)
              [shapefile_dir]/usa_counties/cb_2016_us_county_500k.(...)

    [Outputs]
      - fig: figure object for the user to manipulate after calling this function
      - ax: axes object for the user to manipulate after calling this function

    [References]
        I based my modifications partly on some code snippets in this
        stackoverflow thread:
            https://stackoverflow.com/questions/39742305/how-to-use-basemap-python-to-plot-us-with-50-states
    '''

    from mpl_toolkits.basemap import Basemap as Basemap
    from matplotlib.colors import rgb2hex, Normalize
    from matplotlib.patches import Polygon
    from matplotlib.colorbar import ColorbarBase

    if isinstance(data_per_county,pd.Series):
        data_per_county = data_per_county.to_dict()  # convert to dict
    elif isinstance(data_per_county,pd.DataFrame) and data_per_county.shape[1] == 1:
        data_per_county = data_per_county.iloc[:,0].to_dict()
    elif isinstance(data_per_county,pd.DataFrame) and data_per_county.shape[1] == 2:
        data_per_county = data_per_county.set_index('FIPS_code')  # colunm name hard-coded here for safety
        data_per_county = data_per_county.iloc[:,0].to_dict()

    fig = plt.figure(figsize=(10,7),dpi=dpi)
    ax = plt.gca() # get current axes instance

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(llcrnrlon=-190,llcrnrlat=20,urcrnrlon=-143,urcrnrlat=46,
                projection='merc',lat_ts=20)  # do not change these numbers

    #---------   draw state and county boundaries  ----------------------------
    if shapefile_dir is None:
        shapefile_dir = './shapefiles'
    shp_path_state = os.path.join(shapefile_dir,'usa_states','st99_d00')
    shp_info = m.readshapefile(shp_path_state,'states',drawbounds=True,
                               linewidth=0.45,color='gray')
    shp_info_ = m_.readshapefile(shp_path_state,'states',drawbounds=False)

    cbc = [0.75]*3  # county boundary color
    cbw = 0.15  # county boundary line width
    shp_path_county = os.path.join(shapefile_dir,'usa_counties','cb_2016_us_county_500k')
    shp_info_cnty = m.readshapefile(shp_path_county,'counties',drawbounds=True,
                                    linewidth=cbw,color=cbc)

    shp_info_cnty_ = m_.readshapefile(shp_path_county,'counties',drawbounds=False)

    #-------- choose a color for each county based on unemployment rate -------
    colors={}
    county_FIPS_code_list=[]
    cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(list(data_per_county.values()))
    if vmax is None:
        vmax = np.nanmax(list(data_per_county.values()))
    for shapedict in m.counties_info:
        county_FIPS_code = shapedict['GEOID']
        if county_FIPS_code in data_per_county.keys():
            data_ = data_per_county[county_FIPS_code]
        else:
            data_ = np.nan

        # calling colormap with value between 0 and 1 returns
        # rgba value. Take sqrt root to spread out colors more.
        if not np.isnan(data_):
            colors[county_FIPS_code] = cmap(np.sqrt((data_-vmin)/(vmax-vmin)))[:3]
        else:
            colors[county_FIPS_code] = None

        county_FIPS_code_list.append(county_FIPS_code)

    #---------  cycle through county names, color each one.  --------------------
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for j, seg in enumerate(m.counties):  # for 48 lower states
        shapedict = m.counties_info[j]  # query shape dict at j-th position
        if shapedict['STATEFP'] not in ['02','15']:  # not Alaska or Hawaii
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=color)#,hatch='\\')
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)

    for j, seg in enumerate(m_.counties):  # for Alaska and Hawaii
        shapedict = m.counties_info[j]  # query shape dict at j-th position
        if shapedict['STATEFP']=='02':  # Alaska
            seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y) for x,y in seg]
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)#,hatch='\\')
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)
            ax.add_patch(poly)
        if shapedict['STATEFP']=='15':  # Hawaii
            seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
            if colors[county_FIPS_code_list[j]] == None:
                color = rgb2hex([0.93]*3)
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)#,hatch='\\')
            else:
                color = rgb2hex(colors[county_FIPS_code_list[j]])
                poly = Polygon(seg,facecolor=color,edgecolor=cbc,lw=cbw)
            ax.add_patch(poly)

    ax.set_title(map_title)

    #------------  Plot bounding boxes for Alaska and Hawaii insets  --------------
    light_gray = [0.8]*3  # define light gray color RGB
    m_.plot(np.linspace(170,177),np.linspace(29,29),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(177,180),np.linspace(29,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(180,180),np.linspace(26,23),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-177),np.linspace(23,20),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-180,-175),np.linspace(26,26),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-175,-171),np.linspace(26,22),linewidth=1.,
            color=light_gray,latlon=True)
    m_.plot(np.linspace(-171,-171),np.linspace(22,20),linewidth=1.,
            color=light_gray,latlon=True)

    #------------   Show color bar   ---------------------------------------
    if cmap_midpoint is None:
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=cmap_midpoint)

    ax_c = fig.add_axes([0.92, 0.1, 0.03, 0.8])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical',
                      label=unit)

    mpl_version = mpl.__version__.split('.')
    if float(mpl_version[0] + '.' + mpl_version[0]) >= 2.1: # version > 2.1.0
        cb = adjust_colorbar_tick_labels(cb,
                                         np.nanmax(list(data_per_county.values())) > vmax,
                                         np.nanmin(list(data_per_county.values())) < vmin)

    #------------   Set overall font size  --------------------------------
    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig,ax  # return figure and axes handles

#%%############################################################################
def adjust_colorbar_tick_labels(colorbar_obj,adjust_top=True,adjust_bottom=True):
    '''
    Given a colorbar object (colorbar_obj), change the text of the top (and/or
    bottom) tick label text.

    For example, the top tick label of the color bar is originally "1000", then
    this function change it to ">1000", to represent the cases where the colors
    limits are manually clipped at a certain level (useful for cases with
    extreme values in only some limited locations in the color map).

    Similarly, this function adjusts the lower limit.
    For example, the bottom tick label is originally "0", then this function
    changes it to "<0".

    The second and third parameters control whether or not this function adjusts
    top/bottom labels, and which one(s) to adjust.

    Note: get_ticks() only exists in matplotlib version 2.1.0+, and this function
          does not check for matplotlib version. Use with caution.
    '''

    cbar_ticks = colorbar_obj.get_ticks()  # get_ticks() is only added in ver 2.1.0
    new_ticks = [str(int(a)) if int(a)==a else str(a) for a in cbar_ticks]  # convert to int if possible

    if (adjust_top == True) and (adjust_bottom == True):
        new_ticks[-1] = '>' + new_ticks[-1]   # adjust_top and adjust_bottom may
        new_ticks[0] = '<' + new_ticks[0]     # be numpy.bool_ type, which is
    elif adjust_top == True:                  # different from Python bool type!
        new_ticks[-1] = '>' + new_ticks[-1]   # Thus 'adjust_top == True' is used
    elif adjust_bottom == True:               # here, instead of 'adjust_top is True'.
        new_ticks[0] = '<' + new_ticks[0]
    else:
        pass

    colorbar_obj.ax.set_yticklabels(new_ticks)

    return colorbar_obj

#%%============================================================================
class MidpointNormalize(Normalize):
    '''
    Auxiliary class definition. Copied from:
    https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib/20146989#20146989
    '''
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

#%%============================================================================
def convert_FIPS_to_state_name(dict1):
    '''
    Convert state FIPS codes such as '01' and '45' into full state names.
    '''
    fips2state = {"01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", \
              "08": "CO", "09": "CT", "10": "DE", "11": "DC", "12": "FL", \
              "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN", \
              "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", \
              "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS", \
              "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH", \
              "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND", \
              "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", \
              "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT", \
              "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI", \
              "56": "WY"}  # dictionary mapping FIPS code to state abbreviation

    dict2 = {}  # create empty dict
    for FIPS_code in dict1.keys():
        new_state_name = fips2state[FIPS_code]  # convert state name
        dict2.update({new_state_name: dict1[FIPS_code]})

    dict3 = translate_state_abbrev(dict2,abbrev_to_full=True)

    return dict3

#%%============================================================================
def translate_state_abbrev(dict1,abbrev_to_full=True):
    '''
    Convert state full names into state abbreviations, or the other way.
    '''
    if abbrev_to_full is True:
        translation = {
            'AK': 'Alaska',
            'AL': 'Alabama',
            'AR': 'Arkansas',
            'AS': 'American Samoa',
            'AZ': 'Arizona',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DC': 'District of Columbia',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'GU': 'Guam',
            'HI': 'Hawaii',
            'IA': 'Iowa',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'MA': 'Massachusetts',
            'MD': 'Maryland',
            'ME': 'Maine',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MO': 'Missouri',
            'MP': 'Northern Mariana Islands',
            'MS': 'Mississippi',
            'MT': 'Montana',
            'NA': 'National',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'NE': 'Nebraska',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NV': 'Nevada',
            'NY': 'New York',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'PR': 'Puerto Rico',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VA': 'Virginia',
            'VI': 'Virgin Islands',
            'VT': 'Vermont',
            'WA': 'Washington',
            'WI': 'Wisconsin',
            'WV': 'West Virginia',
            'WY': 'Wyoming'
        }
    else:
        translation = {
            'Alabama': 'AL',
            'Alaska': 'AK',
            'Arizona': 'AZ',
            'Arkansas': 'AR',
            'California': 'CA',
            'Colorado': 'CO',
            'Connecticut': 'CT',
            'Delaware': 'DE',
            'Florida': 'FL',
            'Georgia': 'GA',
            'Hawaii': 'HI',
            'Idaho': 'ID',
            'Illinois': 'IL',
            'Indiana': 'IN',
            'Iowa': 'IA',
            'Kansas': 'KS',
            'Kentucky': 'KY',
            'Louisiana': 'LA',
            'Maine': 'ME',
            'Maryland': 'MD',
            'Massachusetts': 'MA',
            'Michigan': 'MI',
            'Minnesota': 'MN',
            'Mississippi': 'MS',
            'Missouri': 'MO',
            'Montana': 'MT',
            'Nebraska': 'NE',
            'Nevada': 'NV',
            'New Hampshire': 'NH',
            'New Jersey': 'NJ',
            'New Mexico': 'NM',
            'New York': 'NY',
            'North Carolina': 'NC',
            'North Dakota': 'ND',
            'Ohio': 'OH',
            'Oklahoma': 'OK',
            'Oregon': 'OR',
            'Pennsylvania': 'PA',
            'Rhode Island': 'RI',
            'South Carolina': 'SC',
            'South Dakota': 'SD',
            'Tennessee': 'TN',
            'Texas': 'TX',
            'Utah': 'UT',
            'Vermont': 'VT',
            'Virginia': 'VA',
            'Washington': 'WA',
            'West Virginia': 'WV',
            'Wisconsin': 'WI',
            'Wyoming': 'WY',
        }

    dict2 = {}  # create empty dict
    for state_name in dict1.keys():
        new_state_name = translation[state_name]  # convert state name
        dict2.update({new_state_name: dict1[state_name]})

    return dict2

#%%============================================================================
def check_all_states(dict1):
    '''
    Check whether dict1 has all 50 states of USA. If not, append missing state(s)
    to the dictionary and assign np.nan value as its value.

    The state names of dict1 must be full names.
    '''
    full_state_list = [

         'Alabama','Alaska','Arizona','Arkansas','California','Colorado',
         'Connecticut','Delaware','Florida','Georgia','Hawaii','Idaho',
         'Illinois','Indiana','Iowa','Kansas','Kentucky','Louisiana',
         'Maine', 'Maryland','Massachusetts','Michigan','Minnesota',
         'Mississippi', 'Missouri','Montana','Nebraska','Nevada',
         'New Hampshire','New Jersey','New Mexico','New York',
         'North Carolina','North Dakota','Ohio',
         'Oklahoma','Oregon','Pennsylvania','Rhode Island',
         'South Carolina','South Dakota','Tennessee','Texas','Utah',
         'Vermont','Virginia','Washington','West Virginia',
         'Wisconsin','Wyoming'
    ]

    if list(dict1.keys()).sort() != full_state_list:
        dict2 = {}  # create new list
        for state in full_state_list:
            if state in dict1.keys():
                dict2[state] = dict1[state]
            else:
                print('%s data missing (replaced with NaN).'%state)
                dict2[state] = np.nan
    else:
        dict2 = dict1

    return dict2

#%%============================================================================
def plot_timeseries(time_series,fig=None,ax=None,figsize=(10,3),
                   xlabel='Time',ylabel=None,label=None,color=None,lw=2,ls='-',
                   marker=None,fontsize=12,xgrid_on=True,ygrid_on=True,
                   title=None,dpi=96,month_grid_width=None):
    '''
    Plot time_series, where its index indicates a date (e.g., year, month, date).

    [Inputs]
      - time_series: A pandas Series, with index being date;
                     or a pandas DataFrame, with index being date, and each
                     column being a different time series.
      - fig, ax:
            Figure and axes objects.
            If provided, the histograms are plotted on the provided figure and
            axes. If not, a new figure and new axes are created.
      - figsize: figure size
      - xlabel: Label of X axis. Usually "Time" or "Date"
      - ylabel: Label of Y axis. Usually the meaning of the data
      - label:  Label of data, for plotting legends
      - color: Color of line. If None, let Python decide for itself.
      - xgrid_on: Whether or not to show vertical grid lines (default: True)
      - ygrid_on: Whether or not to show horizontal grid lines (default: True)
      - title: Figure title (optional)
      - dpi: Screen resolution
      - month_grid_width: the on-figure "horizontal width" that each time interval
                          occupies. This value determines how X axis labels
                          are displayed (e.g., smaller width leads to date
                          labels being displayed with 90 deg rotation).
                          Do not change this unless you really know what you
                          are doing.

    [Outputs]
      - fig: figure object for the user to manipulate after calling this function
      - ax: axes object for the user to manipulate after calling this function
    '''
    ts = time_series.copy()  # shorten the name + avoid changing some_time_series
    ts.index = map(as_date,ts.index)  # batch-convert index to datetime.date format

    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=figsize,dpi=dpi,facecolor='w',edgecolor='k')
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes()  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    ax.plot(ts.index,ts,color=color,lw=lw,ls=ls,marker=marker,label=label)
    ax.set_label(label)  # set label for legends using argument 'label'
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if month_grid_width == None:
        month_grid_width = figsize[0]/calc_month_interval(ts.index) # width of each month in inches
    ax = format_xlabel(ax,month_grid_width)

    if ygrid_on == True:
        ax.yaxis.grid(ls=':',color=[0.75]*3)
    if xgrid_on == True:
        ax.xaxis.grid(False,'major')
        ax.xaxis.grid(xgrid_on,'minor',ls=':',color=[0.75]*3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax

#%%============================================================================
def plot_multiple_timeseries(multiple_time_series,show_legend=True,
                             figsize=(10,3),dpi=96,**kwargs):
    '''
    This is just a wrapper around plot_timeseries(), which deals with plotting
    multiple time series on the same figure with or without legends.

    I created this function to plot time series for the 50 states in the USA,
    therefore robustness (as well as aesthetics) are not guaranteed for other
    cases.
    '''
    if show_legend is False:  # if no need to show legends, just pass everything
        fig,ax = plot_timeseries(multiple_time_series,**kwargs)  # to plot_timeseries()
    else:
        fig = plt.figure(figsize=figsize,dpi=dpi,facecolor='w',edgecolor='k')
        ax = plt.axes()
        nr_timeseries = multiple_time_series.shape[1]
        linestyle_list = ['-','--','-',':','-','-']
        marker_list = [None,None,'+',None,'o','^']
        if 'marker' in kwargs:
            kwargs.pop('marker')
        if 'ls' in kwargs:
            kwargs.pop('ls')

        for j in range(nr_timeseries):
            plot_timeseries(multiple_time_series.iloc[:,j],
                            fig=fig, ax=ax,
                            label=multiple_time_series.columns[j],
                            ls=linestyle_list[int(j/10) % len(linestyle_list)],
                            marker=marker_list[int(j/10) % len(marker_list)],
                            **kwargs)

        if 'title' not in kwargs:
            bbox_anchor_loc = (0., 1.02, 1., .102)
        else:
            bbox_anchor_loc = (0., 1.08, 1., .102)
        ax.legend(bbox_to_anchor=bbox_anchor_loc,
                  loc='lower center', ncol=10)

    return fig, ax

#%%============================================================================
def fill_timeseries(time_series,upper_bound,lower_bound,
                    fig=None,ax=None,figsize=(10,3),
                    xlabel='Time',ylabel=None,label=None,
                    color=None,lw=3,ls='-',fontsize=12,title=None,dpi=96,
                    xgrid_on=True,ygrid_on=True):
    '''
    Plot time_series as a line, where its index indicates a date (e.g., year,
    month, date).

    And then plot the upper bound and lower bound as shaded areas beneath the line.

    [Inputs]
      - time_series: a pandas Series, with index being date
      - upper_bound, lower_bound: upper/lower bounds of the time series,
                                  must have the same length as time_series
      - fig, ax:
            Figure and axes objects.
            If provided, the histograms are plotted on the provided figure and
            axes. If not, a new figure and new axes are created.
      - figsize: figure size
      - xlabel: Label of X axis. Usually "Time" or "Date"
      - ylabel: Label of Y axis. Usually the meaning of the data
      - label:  Label of data, for plotting legends
      - color: Color of line. If None, let Python decide for itself.
      - lw: line width of the line that represents time_series
      - lw: line style of the line that represents time_series
      - fontsize: font size of the texts in the figure
      - xgrid_on: Whether or not to show vertical grid lines (default: True)
      - ygrid_on: Whether or not to show horizontal grid lines (default: True)
      - title: Figure title (optional)
      - dpi: Screen resolution
      - month_grid_width: the on-figure "horizontal width" that each time interval
                          occupies. This value determines how X axis labels
                          are displayed (e.g., smaller width leads to date
                          labels being displayed with 90 deg rotation).
                          Do not change this unless you really know what you
                          are doing.

    [Outputs]
      - fig: figure object for the user to manipulate after calling this function
      - ax: axes object for the user to manipulate after calling this function
    '''
    ts = time_series.copy()  # shorten the name + avoid changing some_time_series
    ts.index = map(as_date,ts.index)  # batch-convert index to datetime.date format
    lb = lower_bound.copy()
    ub = upper_bound.copy()

    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure(figsize=figsize,dpi=dpi,facecolor='w',edgecolor='k')
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes()  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    ax.fill_between(ts.index,lb,ub,color=color,facecolor=color,
                    linewidth=0.01,alpha=0.5,interpolate=True)
    ax.plot(ts.index,ts,color=color,lw=lw,ls=ls,label=label)
    ax.set_label(label)  # set label for legends using argument 'label'
    ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    month_width = figsize[0]/calc_month_interval(ts.index) # width of each month in inches
    ax = format_xlabel(ax,month_width)

    if ygrid_on == True:
        ax.yaxis.grid(ygrid_on,ls=':',color=[0.75]*3)
    if xgrid_on == True:
        ax.xaxis.grid(False,'major')
        ax.xaxis.grid(xgrid_on,'minor',ls=':',color=[0.75]*3)
    ax.set_axisbelow(True)

    if title is not None:
        ax.set_title(title)

    for o in fig.findobj(mpl.text.Text):
        o.set_fontsize(fontsize)

    return fig, ax

#%%============================================================================
def calc_month_interval(date_array):
    '''
    Calculate how many months are there between the first month and the last
    month of the given date_array.
    '''
    date9 = list(date_array)[-1]
    date0 = list(date_array)[0]
    delta_days = (date9 - date0).days
    delta_months = delta_days//30
    return delta_months

#%%============================================================================
def calc_bar_width(width):
    '''
    Calculate width (in points) of bar plot from figure width (in inches)
    '''
    if width <= 7:
        bar_width = width * 3.35  # these numbers are manually fine-tuned
    elif width <= 9:
        bar_width = width * 2.60
    elif width <= 10:
        bar_width = width * 2.10
    else:
        bar_width = width * 1.2

    return bar_width

#%%============================================================================
def format_xlabel(ax,month_width):
    '''
    Format the X axis label (which represents dates) in accordance to the width
    of each time interval (month or day).

    For narrower cases, year will be put below month.
    For even narrower cases, not every month will be displayed as a label.
    '''
    if month_width < 0.06:
        intvl = 6
    elif month_width < 0.09:
        intvl = 4
    elif month_width < 0.15:
        intvl = 3
    elif month_width < 0.25:
        intvl = 2
    else:
        intvl = 1

    years = mpl.dates.YearLocator()
    months = mpl.dates.MonthLocator(interval=intvl)  # only show every two months
    years_fmt = mpl.dates.DateFormatter('\n%Y')  # add some space for the year label
    months_fmt = mpl.dates.DateFormatter('%m')

    ax.xaxis.set_minor_locator(months)
    #ax.xaxis.set_minor_locator(tkr.MultipleLocator(base=60))  # date interval: 60 days (~2 months)
    ax.xaxis.set_minor_formatter(months_fmt)
    #if absolute_bar_width <= 0.25:  # for narrow bars
    #    plt.setp(ax.xaxis.get_minorticklabels(),rotation=90)  # rotate 'month' label 90 degrees
    #else:  # for wider bars
    #    pass  # do not rotate (i.e., rotation = 0)
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(years_fmt)
    ax.tick_params(labelright=True)  # also show y axis on right edge of figure
    return ax

#%%============================================================================
def as_date(str_date):
    '''
    Convert string date to datetime array.

    It can handle:
    (A) A list of str, int, or float, such as:
        [1] ['20150101', '20150201', '20160101']
        [2] ['2015-01-01', '2015-02-01', '2016-01-01']
        [3] [201405, 201406, 201407]
        [4] [201405.0, 201406.0, 201407.0]
    (B) A list of just a single element, such as:
        [1] [201405]
        [2] ['2014-05-25']
        [3] [201412.0]
    (C) A single element of: str, int, float, such as:
        [1] 201310
        [2] 201210.0
    (D) A pandas Series, of length 1 or length larger than 1

    Firstly, this function does some parsing (e.g., determining whether str_date is
    a list of a single element, or a list of multiple element, or just a single element),
    and then this function calls str2date_kernel() to do the conversion to a single element.

    Reference: https://docs.python.org/2/library/datetime.html#strftime-strptime-behavior
    '''
    #print(type(str_date))

    if pd.__version__ == '0.17.1':
        timestamp_type = pd.tslib.Timestamp
    else:
        timestamp_type = pd._libs.tslib.Timestamp

    if isinstance(str_date,timestamp_type):  # if already a pandas Timestamp obj
        date_list = str_date  # return str_date as is
    else:
        # -----------  Convert to list for pd.Series or np.ndarray objects  -------
        if isinstance(str_date,pd.Series):
            str_date = list(str_date)
        if isinstance(str_date,np.ndarray):
            str_date = list(str_date)

        # ----------  Element-wise checks and conversion  -------------------------
        if isinstance(str_date,list):   # if input is a list
            if len(str_date) == 0:  # empty list
                date_list = None   # return an empty object
            elif len(str_date) == 1:  # length of string is 1
                date_ = str(int(str_date[0]))  # simply unpack it and convert to str int
                date_list = str2date_kernel(date_)
            else:  # length is larger than 1
                nr = len(str_date)
                date_list = [[None]] * nr
                for j in range(nr):
                    if isinstance(str_date[j],str) and str_date[j].isdigit():
                        date_ = str(int(str_date[j]))
                    if isinstance(str_date[j],str) and not str_date[j].isdigit():
                        date_ = str_date[j]
                    if isinstance(str_date[j],int) or isinstance(str_date[j],np.float64):
                        date_ = str(int(str_date[j]))  # robustness not guarenteed!
                    date_list[j] = str2date_kernel(date_)  # do conversion element by element
        elif type(str_date) == dt.date:  # if a datetime.date object
            date_list = str_date  # no need for conversion
        elif isinstance(str_date,int) or isinstance(str_date,np.float64):  # integer or float
            date_ = str(int(str_date))
            date_list = str2date_kernel(date_)
        elif isinstance(str_date,str):  # a single string, such as '2015-04'
            date_ = str_date  # no conversion needed
            date_list = str2date_kernel(date_)
        else:
            print('#####  Edge case encountered! (Input data type of str_date not recognized.) #####')
            print('\ntype(str_date) is: %s' % type(str_date))
            try:
                print('Length of str_date is: %s' % len(str_date))
            except TypeError:
                print('str_date has no length.')

    return date_list

#%%============================================================================
def str2date_kernel(date_):
    '''
    Convert date_ into a datetime object. date_ must be a string (not a list of strings).

    Currently accepted date formats:
    (1) Aug-2014
    (2) August 2014
    (3) 201407
    (4) 2016-07
    (5) 2015-02-21
    '''
    import datetime as dt

    day = None
    if ('-' in date_) and (len(date_) == 8):  # for date style 'Aug-2014'
        month, year = date_.split('-')  # split string by character
        month = dt.datetime.strptime(month,'%b').month  # from 'Mar' to '3'
    elif ' ' in date_:  # for date style 'August 2014'
        month, year = date_.split(' ')  # split string by character
        month = dt.datetime.strptime(month,'%B').month  # from 'March' to '3'
        year = int(year)
    elif (len(date_) == 6) and date_.isdigit():  # for cases like '201205'
        year  = int(date_[:4])  # first four characters
        month = int(date_[4:])  # remaining characters
    elif (len(date_) == 7) and (date_[4]=='-') and not date_.isdigit():  # such as '2015-03' [NOT 100% ROBUST!]
        year, month = date_.split('-')
        year = int(year)
        month = int(month)
    elif (len(date_) == 10) and not date_.isdigit():  # such as '2012-02-01' [NOT 100% ROBUST!!]
        year, month, day = date_.split('-')  # split string by character
        year = int(year)
        month = int(month)
        day = int(day)
    elif (len(date_)==6) and (date_[3]=='-') and (date_[:3].isalpha()) \
         and (date_[4:].isdigit()):  # such as 'May-12'
        month, year = date_.split('-')
        month = dt.datetime.strptime(month,'%b').month  # from 'Mar' to '3'
        year = int(year) + 2000  # from '13' to '2013'
    else:
        print('*****  Edge case encountered! (Date format not recognized.)  *****')
        print('\nUser supplied %s, which is not recognized.\n' % date_)

    if day is None:  # if day is not defined in the if statements
        return dt.date(year,month,1)
    else:
        return dt.date(year,month,day)

#%%============================================================================
def plot_with_error_bounds(x,y,upper_bound,lower_bound,line_color=[0.4]*3,
                           shade_color=[0.7]*3,shade_alpha=0.5,linewidth=2.0,
                           legend_loc='best',
                           line_label='Data',shade_label='$\mathregular{\pm}$STD',
                           fig=None,ax=None,
                           x_scale='linear',y_scale='linear',grid_on=True):
    '''
    Plot a graph with one line and its upper and lower bounds, with areas between
    bounds shaded. The effect is similar to this illustration below.


      y ^            ...                         ______________________
        |         ...   ..........              |                     |
        |         .   ______     .              |  ---  Mean value    |
        |      ...   /      \    ..             |  ...  Error bounds  |
        |   ...  ___/        \    ...           |_____________________|
        |  .    /    ...      \    ........
        | .  __/   ...  ....   \________  .
        |  /    ....       ...          \
        | /  ....            .....       \_
        | ...                    ..........
       -|--------------------------------------->  x


    [Inputs]
        x, y: data points to be plotted as a line (should have the same length)
        upper_bound, lower_bound: Upper and lower bounds to be plotted as
                                  shaded areas. Should have the same length as
                                  x and y.
        line_color: color of the line of y
        shade_color: color of the underlying shades
        shade_alpha: transparency of the shades
        linewidth: width of the line of y
        legend_loc: location of the legend, to be passed directly to plt.legend()
        line_label: label of the line of y, to be used in the legend
        shade_label: label of the shades, to be used in the legend
        fig, ax:
            Figure and axes objects.
            If provided, the histograms are plotted on the provided figure and
            axes. If not, a new figure and new axes are created.        y_scale
        grid_on: whether or not to show the grids

    [Outputs]
        fig: figure object for the user to manipulate after calling this function
        ax: axes object for the user to manipulate after calling this function
    '''
    if fig is None:  # if a figure handle is not provided, create new figure
        fig = pl.figure()
    else:   # if provided, plot to the specified figure
        pl.figure(fig.number)

    if ax is None:  # if ax is not provided
        ax = plt.axes()  # create new axes and plot lines on it
    else:
        ax = ax  # plot lines on the provided axes handle

    hl1 = ax.fill_between(x, lower_bound, upper_bound,
                           color=shade_color, facecolor=shade_color,
                           linewidth=0.01, alpha=shade_alpha, interpolate=True,
                           label=shade_label)
    hl2, = ax.plot(x, y, color=line_color, linewidth=linewidth, label=line_label)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)

    if grid_on == True:
        ax.grid(ls=':',lw=0.5)
        ax.set_axisbelow(True)

    plt.legend(handles=[hl2,hl1],loc=legend_loc)

    return (fig, ax)

