import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for projection='3d'


def get_ax3d(figure=None, subplot=111, figsize=(4, 3), elev=45, azim=45):
    """
    Get axis for a 3d plot
    :param figure: figure handle or open a new figure with None
    :param figsize:
    :param elev: elevation of camera view (degree)
    :param azim: azimut value of camera view (degree)
    :return:
    """
    if not figure:
        figure = plt.figure(figsize=figsize)
    ax = figure.add_subplot(subplot, projection='3d')
    ax.view_init(elev, azim)
    return ax
