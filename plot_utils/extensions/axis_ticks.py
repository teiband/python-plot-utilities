def invert_axis_ticks(ax, axis):
    """
    Invert the ticks on a specified axis
    :param ax: AxisObject (pyplot)
    :param axis: 'x' or 'y'
    :return:
    """
    if axis == 'x':
        ax.set_xticklabels(ax.get_xticks()[::-1])
    elif axis == 'y':
        ax.set_yticklabels(ax.get_yticks()[::-1])
    else:
        raise ValueError
