def create_legend_outside(ax, labels, pos='bottom', ratio=0.2, **kwargs):
    """
    param pos: Put legend at bottom or left or right of plot
    param ratio: Ratio of space the legend takes on the plot
    """
    box = ax.get_position()
    if pos == 'bottom':
        # Shrink current axis's height by 10% on the bottom
        ax.set_position([box.x0, box.y0 + box.height * ratio,
                         box.width, box.height * (1 - ratio)])
        bbox = (0.5, -0.3)
        loc = 'center'
        ncol = len(labels)
    elif pos == 'left':
        ax.set_position([box.x0 + box.width * ratio, box.y0,
                         box.width * (1 - ratio), box.height])
        bbox = (-ratio, 0.5)
        loc = 'center right'
        ncol = 1
    elif pos == 'right':
        ax.set_position([box.x0, box.y0,
                         box.width * (1 - ratio), box.height])
        bbox = (1, 0.5)
        loc = 'center left'
        ncol = 1
    else:
        raise NotImplementedError
    lh = ax.legend(labels, loc=loc, bbox_to_anchor=bbox, ncol=ncol, **kwargs)
    # leg.get_frame().set_linewidth(0.0)
    return lh
