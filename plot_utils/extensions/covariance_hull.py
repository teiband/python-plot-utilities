import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse


def plot_mu_and_sigma(ax, x, mu, sigma, c_mu='b', c_sigma=[0.85, 0.85, 1]):
    ax.plot(x, mu, c_mu)
    ax.fill_between(x, mu + 0.5 * sigma, mu - 0.5 * sigma, color=c_sigma)


def construct_cov_hull(mus, covs, ax, color=[0.1, 0.5, 0.3, 0.05], label=None):
    for i, (mu, cov) in enumerate(zip(mus, covs)):
        if not i % 10:  # take every nth sample
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]

            # rx, ry, rz = 1/np.sqrt(coefs)
            rx, ry, rz = 0.05 * np.sqrt(eigvals)  # get standard deviation from variance

            # Set of all spherical angles:
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)

            # Cartesian coordinates that correspond to the spherical angles:
            # (this is the equation of an ellipsoid):
            x = rx * np.outer(np.cos(u), np.sin(v))
            y = ry * np.outer(np.sin(u), np.sin(v))
            z = rz * np.outer(np.ones_like(u), np.cos(v))

            points = np.array([x, y, z]).T

            # rotate around origin
            from scipy.spatial.transform import Rotation
            R = eigvecs
            points_flat = points.reshape(100, 3)
            points_flat = Rotation.from_dcm(R).apply(points_flat)
            points = points_flat.reshape(10, 10, 3)

            # shift by means
            points += mu

            # Plot:
            ax.plot_surface(*points.T, rstride=4, cstride=4, color=color, label=label)


# from: https://gist.github.com/superjax/80fefc02551b42ee6a70fc85f287a73c
def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip


if __name__ == '__main__':
    # -- Example usage -----------------------
    # Generate some random, correlated data
    points = np.random.multivariate_normal(
        mean=(1, 1), cov=[[0.4, 9], [9, 10]], size=1000
    )
    # Plot the raw points...
    x, y = points.T
    plt.plot(x, y, 'ro')

    # Plot a transparent 3 standard deviation covariance ellipse
    plot_point_cov(points, nstd=3, alpha=0.5, color='green')

    plt.show()
