# plot_utils.choropleth_map_state

**plot_utils.choropleth_map_state**(*data_per_state, vmin=None, vmax=None, map_title='USA map', unit='', cmap='hot_r', fontsize=14, cmap_midpoint=None, figsize=(10,7), dpi=100, shapefile_dir=None*):

Generate a choropleth map of USA (including Alaska and Hawaii), on a state
level.

According to wikipedia, a choropleth map is a thematic map in which areas
are shaded or patterned in proportion to the measurement of the statistical
variable being displayed on the map, such as population density or
per-capita income.

#### [Parameters]
    data_per_state:
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
    vmin:
        Minimum value to be shown on the map. If vmin is larger than the
        actual minimum value in the data, some of the data values will be
        "clipped". This is useful if there are extreme values in the data
        and you do not want those values to complete skew the color
        distribution.
    vmax:
        Maximum value to be shown on the map. Similar to vmin.
    map_title:
        Title of the map, to be shown on the top of the map.
    unit:
        Unit of the numerical (for example, "population per km^2"), to be
        shown on the right side of the color bar.
    cmap:
        Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
        for plotting deviation maps.
    fontsize:
        Font size of all the texts on the map.
    cmap_midpoint:
        A numerical value that specifies the "deviation point". For example,
        if your data ranges from -200 to 1000, and you want negative values
        to appear blue-ish, and positive values to appear red-ish, then you
        can set cmap_midpoint to 0.0.
    figsize:
        Size (width,height) of figure (including map and color bar).
    dpi:
        On-screen resolution.
    shapefile_dir:
        Directory where shape files are stored. Shape files (state level and
        county level) should be organized as follows:
            [shapefile_dir]/usa_states/st99_d00.(...)
            [shapefile_dir]/usa_counties/cb_2016_us_county_500k.(...)

#### [Returns]
    fix, ax:
        Figure and axes objects

#### [References]
    I based my modifications partly on some code snippets in this stackoverflow thread: https://stackoverflow.com/questions/39742305

-------------------------------------------------------
    
# plot_utils.choropleth_map_county

**plot_utils.choropleth_map_county**(*data_per_county, vmin=None, vmax=None, unit='', cmap='hot_r', map_title='USA county map', fontsize=14, cmap_midpoint=None, figsize=(10,7), dpi=100, shapefile_dir=None*):

Generate a choropleth map of USA (including Alaska and Hawaii), on a county level.

According to wikipedia, a choropleth map is a thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map, such as population density or per-capita income.

#### [Parameters]
    data_per_county:
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
    vmin:
        Minimum value to be shown on the map. If vmin is larger than the
        actual minimum value in the data, some of the data values will be
        "clipped". This is useful if there are extreme values in the data
        and you do not want those values to complete skew the color
        distribution.
    vmax:
        Maximum value to be shown on the map. Similar to vmin.
    map_title:
        Title of the map, to be shown on the top of the map.
    unit:
        Unit of the numerical (for example, "population per km^2"), to be
        shown on the right side of the color bar.
    cmap:
        Color map name. Suggested names: 'hot_r', 'summer_r', and 'RdYlBu'
        for plotting deviation maps.
    fontsize:
        Font size of all the texts on the map.
    cmap_midpoint:
        A numerical value that specifies the "deviation point". For example,
        if your data ranges from -200 to 1000, and you want negative values
        to appear blue-ish, and positive values to appear red-ish, then you
        can set cmap_midpoint to 0.0.
    figsize:
        Size (width, height) of figure (including map and color bar).
    dpi:
        On-screen resolution.
    shapefile_dir:
        Directory where shape files are stored. Shape files (state level and
        county level) should be organized as follows:
            [shapefile_dir]/usa_states/st99_d00.(...)
            [shapefile_dir]/usa_counties/cb_2016_us_county_500k.(...)

#### [Returns]
    fix, ax:
        Figure and axes objects

#### [References]
    I based my modifications partly on some code snippets in this
    stackoverflow thread: https://stackoverflow.com/questions/39742305
