import matplotlib.axis
import matplotlib.pyplot as plt


def decorate(handle, xlabel, ylabel, legend_loc=None, legend_ncol=None, legend_bbox_to_anchor=None):
    """
    Decorate a plot axis with labels and legend.
    The labels are put as latex math symbols, e.g. you can provide "\alpha" or just "t"
    :param handle:
    :param xlabel: string or latex math expression
    :param ylabel: string or latex math expression
    :param legend_loc:
    :param legend_ncol:
    :param legend_bbox_to_anchor:
    :param elev: elevation of camera on 3d plot
    :param legend_bbox_to_anchor:
    :return:
    """
    if isinstance(handle, plt.__class__):
        ax = plt.gca()
    elif isinstance(handle, matplotlib.axis):
        ax = handle
    else:
        raise NotImplementedError
    legend_settings = {}
    if legend_loc:
        legend_settings['loc'] = legend_loc
    if legend_ncol:
        legend_settings['ncol'] = legend_ncol
    if legend_bbox_to_anchor:
        legend_settings['bbox_to_anchor'] = legend_bbox_to_anchor
    # legend_settings = {'loc': 'lower left', 'ncol': 1, 'bbox_to_anchor': (0, -0.15)}
    ax.legend(**legend_settings)
    ax.grid()
    ax.set_xlabel('$' + xlabel + '$')
    ax.set_ylabel('$' + ylabel + '$', rotation=0)
    plt.tight_layout(pad=1)
