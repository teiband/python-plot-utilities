# plot_utils.Color

A class that defines a color.

#### Public attriutes

    None.

#### Initialization

    Color(color, is_rgb_normalized=True)

#### Public methods

    Color.as_rgb(normalize=True):
        Export the color object as RGB values (a tuple).

    Color.as_rgba(alpha=1.0):
        Export the color object as RGBA values (a tuple).

    Color.as_hex():
        Export the color object as HEX values (a string).

    Color.show():
        Show color as a square patch.
        
-------------------------------------------------------

# plot_utils.Multiple_Colors

A class that defines multiple colors.

#### Public attriutes

    None.

#### Initialization

    Multiple_Colors(colors, is_rgb_normalized=True)

#### Public methods

    Multiple_Colors.as_rgb(normalize=True):
        Export the colors as a list of RGB values.

    Multiple_Colors.as_rgba(alpha=1.0):
        Export the colors as a list of RGBA values.

    Multiple_Colors.as_hex():
        Export the colors as a list of HEX values.

    Multiple_Colors.show(vertical=False):
        Show colors as square patches.
