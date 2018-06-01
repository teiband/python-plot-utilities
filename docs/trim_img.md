# plot_utils.trim_img

**plot_utils.trim_img**(*files, pad_width=20, pad_color='w', inplace=False, verbose=True, show_old_img=False, show_new_img=False, forcibly_overwrite=False*):

Trim the margins of image file(s) on the hard drive, and (optionally) add padded margins of a specified color and width.

#### [Parameters]
    
    files : <str> or <list, tuple>
        A file name (as Python str) or several file names (as Python list or
        tuple) to be trimmed.
    pad_width : <float>
        The amount of white margins to be padded (unit: pixels). Float pad_width
        values are internally converted as int.
    pad_color : <str> or <tuple> or <list>
        The color of the padded margin. Valid pad_color values are color names
        recognizable by matplotlib: https://matplotlib.org/tutorials/colors/colors.html
    inplace : <bool>
        Whether or not to replace the existing figure file with the trimmed
        content.
    verbose : <bool>
        Whether or not to print the progress onto the console.
    show_old_img : <bool>
        Whether or not to show the old figure in the console.
    show_new_img : <bool>
        Whether or not to show the trimmed figure in the  console

#### [Returns]
    
    None