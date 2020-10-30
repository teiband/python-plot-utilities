from scipy.spatial.transform.rotation import Rotation


def cuboid_data(center, size):
    """
    Create a data array for cuboid plotting.
    # https://stackoverflow.com/questions/30715083/python-plotting-a-wireframe-3d-cuboid


    ============= ================================================
    Argument      Description
    ============= ================================================
    center        center of the cuboid, triple
    size          size of the cuboid, triple, (x_length,y_width,z_height)
    :type size: tuple, numpy.array, list
    :param size: size of the cuboid, triple, (x_length,y_width,z_height)
    :type center: tuple, numpy.array, list
    :param center: center of the cuboid, triple, (x,y,z)
    """

    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],  # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]  # y coordinate of points in inside surface
    z = [[o[2], o[2], o[2], o[2], o[2]],  # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],  # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],  # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]  # z coordinate of points in inside surface
    return x, y, z


def plot_cube_at(pos=(0, 0, 0), size=(1, 1, 1), ax=None, **kwargs):
    # Plotting a cube element at position pos
    if ax != None:
        X, Y, Z = cuboid_data(pos, size)
        ax.plot_surface(X, Y, np.array(Z), rstride=1, cstride=1, **kwargs)


def plot_cube_at_pos_orientation(pos=(0, 0, 0), size=(1, 1, 1), orientation=(0, 0, 0), ax=None, **kwargs):
    """
    Plot a cube/cuboid at any place, with any size and with any orientation
    :param pos:
    :param size:
    :param orientation: as rotation matrix or euler angles (XYZ) in degrees
    :param ax:
    :param kwargs:
    :return:
    """
    if ax != None:
        X, Y, Z = cuboid_data(pos, size)
        # rotate coordinates
        if len(orientation) == 9:
            R = Rotation.from_dcm(orientation)
        elif len(orientation) == 3:
            R = Rotation.from_euler('XYZ', orientation, degrees=True)
        else:
            raise ValueError

        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        points = np.hstack((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)))

        # shift to origin
        points -= np.array(pos)

        # rotate
        points = R.apply(points)

        # shift back
        points += np.array(pos)

        X = points[:, 0].reshape(4, 5)
        Y = points[:, 1].reshape(4, 5)
        Z = points[:, 2].reshape(4, 5)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, **kwargs)
