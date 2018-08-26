# Python plotting utilities: `plot_utils`

This is a Python module that contains some useful data visualization tools.

Current functionalities include (for full list, use `print(plot_utils.__doc__)`):

#### 1. Visualizing one column of data

- **Pie chart**: proportions of distinct values in an array, more convenient than matplotlib's `pie()` function [[doc](./docs/piechart.md)], [[example](./examples/Pie_chart_example.ipynb)]
- **Discrete histogram**: counts of distinct values in an array [[doc](./docs/discrete_histogram.md)], [[example](./examples/Discrete_histogram_example.ipynb)]

#### 2. Visualizing two columns of data ([[doc](./docs/two_columns.md)], [[example](./examples/Two_columns_of_data_example.ipynb)])

- **"Bin-and-mean" plot**: for two continuous variables
- **Category mean**: for a categorical variable and a continuous variable
- **Positive rate**: for a categorical variable and a binary categorical variable
- **Contingency table**: for two categorical variables

#### 3. Visualizing multiple columns of data

+ **3D histograms**: distributions of multiple variables [[doc](./docs/histogram3d.md)], [[example](./examples/3D_histograms_example.ipynb)]
+ **Violin plot**: distribution of multiple variables 
+ **Correlation matrix**: correlation between each columns of a dataset [[doc](./docs/plot_correlation.md)], [[example](./examples/Correlation_matrix_examples.ipynb)]
    - and the one-to-one **scatter plots** for the variables within the dataset [[doc](./docs/scatter_plots_two_cols.md)]
+ **Count missing values**: how many missing values are there in each column of the dataset [[doc](./docs/missing_value_counts.md)], [[example](./examples/Missing_value_count_example.ipynb)]


#### 4. Map plotting

+ **Choropleth map** (a.k.a., "heat map") of the United States, on both the state and county level [[doc](./docs/choropleth_map.md)], [[example](./examples/Choropleth_map_example.ipynb)]


#### 5. Time series plotting

- **Plot single time series**: [[doc](./docs/plot_timeseries.md)], [[example](./examples/Plot_time_series_example.ipynb)]
- **Plot multiple time series**: [[doc](./docs/plot_timeseries.md)], [[example](./examples/Plot_time_series_example.ipynb)]

#### 6. Miscellaneous

+ A **get_colors()** function that conveniently queries different color palettes [[doc](./docs/get_colors.md)], [[example](./examples/Color_and_linespec_examples.ipynb)]
+ A **get_linespecs()** function that generates distinct color/linestyle/linewidth combinations for plotting many lines together [[doc](./docs/get_linespecs.md)], [[example](./examples/Color_and_linespec_examples.ipynb)]
+ Two helper classes: **Color** and **Multiple_Colors**, which make querying and displaying colors more easily  [[doc](./docs/color_classes.md)], [[example](./examples/Color_and_linespec_examples.ipynb)]
+ **Plotting with upper/lower error bounds**, which displays error bounds as shaded areas [[doc](./docs/plot_with_error_bounds.md)], [[example](./examples/Plot_with_error_bounds_example.ipynb)]
+ **trim_img()**, which trims the white margins of the specified image file(s) [[doc](./docs/trim_img.md)]




## Gallery

### 1. One column of data

```python
import pandas as pd
import plot_utils as pu
titanic = pd.read_csv('./examples/datasets/titanic3.csv')
pu.piechart(titanic['survived'], title='Suvived')
pu.discrete_histogram(titanic['pclass'], xlabel='Passenger ticket class')
```

Piechart: [[doc](./docs/piechart.md)], [[example](./examples/Pie_chart_example.ipynb)] 

Discrete histogram: [[doc](./docs/discrete_histogram.md)], [[example](./examples/Discrete_histogram_example.ipynb)]

![](./examples/gallery/piechart_and_discrete_hist.png)



### 2. Two columns of data

```python
titanic = pd.read_csv('./examples/datasets/titanic3.csv')
titanic.rename(index=str, columns={'pclass':'ticket_class'}, inplace=True)
titanic['embarked'] = titanic['embarked'].map({'S':'Southampton',
                                               'C':'Cherbourg',
                                               'Q':'Queenstown'})

pu.bin_and_mean(titanic['age'], titanic['fare'], figsize=(5,3))
pu.category_means(titanic['ticket_class'], titanic['fare'])
pu.positive_rate(titanic['ticket_class'], titanic['survived'], figsize=(5,2))
pu.contingency_table(titanic['ticket_class'], titanic['embarked'], dropna=True, rot=0)
```

[[doc](./docs/two_columns.md)], [[example](./examples/Two_columns_of_data_example.ipynb)] 

![](./examples/gallery/two_variables.png)



### 3. Multiple columns of data

#### 3.1. 3D histograms

Useful for comparing multiple distributions.

```python
iris = pd.read_csv('./examples/datasets/iris.csv')
pu.histogram3d(iris[['petal_width', 'petal_length', 'sepal_width', 'sepal_length']])
```

[[doc](./docs/histogram3d.md)], [[example](./examples/3D_histograms_example.ipynb)]

![histogram_3d](./examples/gallery/histogram_3d.png)



#### 3.2. Violin plots

Another useful tool to compare multiple distributions.

```python
pu.violin_plot(iris[['petal_width', 'petal_length', 'sepal_width', 'sepal_length']])
```

[[doc](./docs/violin_plot.md)], [[example](./examples/Violin_plot_example.ipynb)]

![](./examples/gallery/violin_plots.png)



#### 3.2. Correlation matrix

```python
iris = pd.read_csv('./examples/datasets/iris.csv')
pu.plot_correlation(iris, scatter_plots=True)
```

The first figure shows the correlation (or "sample covariance")  between each column. The second figure shows the scatter plots between each pair of columns.

[[doc](./docs/plot_correlation.md)], [[example](./examples/Correlation_matrix_examples.ipynb)]

![](./examples/gallery/correlation_matrix.png)

![](./examples/gallery/scatter_plots.png)



#### 3.3. Count missing values

```python
titanic = pd.read_csv('./examples/datasets/titanic3.csv')
pu.missing_value_counts(titanic)
```

[[doc](./docs/missing_value_counts.md)], [[example](./examples/Missing_value_count_example.ipynb)]

Each bar corresponds to a column in `titanic`, and the numbers atop are the missing data counts for the corresponding column.

![](./examples/gallery/missing_values.png)



### 4. Choropleth maps (a.k.a., "heat maps")

#### 4.1. State-level choropleth maps for the US 

```python
pu.choropleth_map_state(state_level_data)  # see [example] for details of "state_level_data"
```

[[doc](./docs/choropleth_map.md)], [[example](./examples/Choropleth_map_example.ipynb)]

![choropleth_map_state](./examples/gallery/choropleth_map_state.png)

#### 4.2. County-level choropleth maps for the US

```python
pu.choropleth_map_county(county_level_data)  # see [example] for details of "county_level_data"
```

[[doc](./docs/choropleth_map.md#plot_utilschoropleth_map_county)], [[example](./examples/Choropleth_map_example.ipynb)]

![choropleth_map_county](./examples/gallery/choropleth_map_county.png)

### 5. Time series plotting

#### 5.1. Single time series

```python
df = pd.read_csv('./examples/datasets/Unemployment_rate_1976-2017.csv', index_col=0)
pu.plot_timeseries(df['CA'], ylabel='Unit: %', title='Unemployment rate, California')
```

[[doc](./docs/plot_timeseries.md)], [[example](./examples/Plot_time_series_example.ipynb)]

![](./examples/gallery/time_series_single.png)

#### 5.2. Multiple time series

```python
pu.plot_multiple_timeseries(df, ylabel='Unemployment rate [%]', ncol_legend=10)
```

[[doc](./docs/plot_timeseries.md)], [[example](./examples/Plot_time_series_example.ipynb)]

![](./examples/gallery/time_series.png)



### 6. Miscellaneous

#### 6.1. `get_colors()` function

Easy querying of distinguishable color palettes.

```python
colors = pu.get_colors(color_scheme='tab10', N=10)  # a Python list containing 10 colors
pu.Multiple_Colors(colors).show()  # show colors as a palette
```

[[doc](./docs/get_colors.md)], [[example](./examples/Color_and_linespec_examples.ipynb)]

![](./examples/gallery/get_colors.png)

#### 6.2. `get_linespecs()` function

Easy querying of distinguishable line specs.

```python
line_specs = pu.get_linespecs(color_scheme='bw',range_linewidth=[3,8],priority='linewidth')
pu.linespecs_demo(line_specs)
```

[[doc](./docs/get_linespecs.md)], [[example](./examples/Color_and_linespec_examples.ipynb)]

![](./examples/gallery/get_linespecs.png)

#### 6.3. Plot with error bounds

Plots data and error bounds on the same graph.

```python
pu.plot_with_error_bounds(data, upper_bound, lower_bound)
```

[[doc](./docs/plot_with_error_bounds.md)], [[example](./examples/Plot_with_error_bounds_example.ipynb)]

![](./examples/gallery/error_bounds.png)



## Installation

##### 1. Install with `pip` (recommended):

`>>> pip install git+https://github.com/jsh9/python-plot-utilities`

##### 2. The portable way:

Just download this repository, and you can put `plot_utils.py` anywhere within your Python search path.

#####Note:

If you run into the following issue on Mac OS X (or macOS) when importing `plot_utils`:

`RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework.`

please follow this solution to fix the issue: https://stackoverflow.com/a/21789908/8892243



## Dependencies

+ Python 2.7 or 3.5+
+ matplotlib 1.5.0+, or 2.0.0+ (Version 2.1.0+ is strongly recommended.)
+ numpy: 1.11.0+
+ scipy: 0.19.0+
+ pandas: 0.20.0+
+ cycler: 0.10.0+
+ matplotlib/basemap: 1.0.7 (only if you want to plot the two choropleth maps)
+ PIL (only if you want to use `trim_img`)




## Aesthetics

The aesthetics of of the `plot_utils` module are matplotlib-styled by default, but it doesn't mean that you can't use your favorite styles in seaborn or ggplot2.

Unlike some plotting packages that enforces their own styles and restrict users from customizing, users of this module can adjust the figure styles freely: either from within matplotlib (https://matplotlib.org/devdocs/gallery/style_sheets/style_sheets_reference.html), or `import seaborn` and let seaborn take care of everything.



## References

I did not built every function of this module entirely from scratch. I documented the sources that I referenced in the documentation of the corresponding functions.



## Copyright and license

(c) 2017-2018, Jian Shi

License: GPL v3.0
