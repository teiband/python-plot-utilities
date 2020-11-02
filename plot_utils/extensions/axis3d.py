import matplotlib.pyplot as plt


def get_ax3d(new_figure=True, figsize=(4, 3), elev=45, azim=45):
    """
    Get axis for a 3d plot
    :param new_figure: open new figure
    :param figsize:
    :param elev: elevation of camera view (degree)
    :param azim: azimut value of camera view (degree)
    :return:
    """
    if new_figure:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev, azim)
    return ax
