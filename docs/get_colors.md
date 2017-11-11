# plot_utils.get_colors

**plot_utils.get_colors**(*N,color_scheme='tab10'*):

Returns a list of N distinguisable colors. When N is larger than the color scheme capacity, the color cycle is wrapped around.

What does each color_scheme look like?
    https://matplotlib.org/mpl_examples/color/colormaps_reference_04.png
    https://matplotlib.org/users/dflt_style_changes.html#colors-color-cycles-and-color-maps
    https://github.com/vega/vega/wiki/Scales#scale-range-literals
    https://www.mathworks.com/help/matlab/graphics_transition/why-are-plot-lines-different-colors.html

#### [Parameters]
    N : <int> or None
        Number of qualitative colors desired. If None, returns all the colors
        in the specified color scheme.
    color_scheme : <str> or {0, 10, 8.3, 8.4}
        Color scheme specifier. Valid specifiers are:
        (1) Matplotlib qualitative color map names:
            'Pastel1'
            'Pastel2'
            'Paired'
            'Accent'
            'Dark2'
            'Set1'
            'Set2'
            'Set3'
            'tab10'
            'tab20'
            'tab20b'
            'tab20c'
![](https://matplotlib.org/mpl_examples/color/colormaps_reference_04.png)
        (2) 8.3 and 8.4 (floats): old and new MATLAB color scheme
            Old:
![](https://www.mathworks.com/help/matlab/graphics_transition/transition_colororder_old.png)
            New:
![](https://www.mathworks.com/help/matlab/graphics_transition/transition_colororder.png)
        (3) 0 and 10 (ints):
            0 is equivalent to old MATLAB scheme; 10 is equivalent to 'tab10'.

#### [Returns]
    A list of colors.
