from itertools import product
import numpy as np

runif = np.random.uniform


def gen_random_ellipse_wts(shp, blobs=1):
    """
    Generate a random Weights image as a gaussian ellipse with a reasonable
    mean vector and covariance matrix.
    shp:    integer tuple specifying shape of the image
    blobs:  number of ellipses dumped on top of each other
    """
    from numpy import linalg as la
    wt = np.zeros(shp)
    for b in range(blobs):
        # Covairance
        sig = [runif(dim / 8., dim / 4.) for dim in shp]
        cov = sig[0] * sig[1] * runif(-1, 1)
        cov_mat = np.matrix([[sig[0] ** 2, cov], [cov, sig[1] ** 2]])

        # Mean
        mu = np.array([runif(dim / 8., 7 * dim / 8.) for dim in shp])

        # Fill the matrix
        for pt in product(range(shp[0]), range(shp[1])):
            wt[pt] += np.exp(-.5 * la.solve(cov_mat, pt - mu).dot(pt - mu))

    wt -= wt.mean()
    return wt.flatten()


def gabor(size, alpha, freq, sgm, center=None, phi=0, res=1, ampl=1.):
    """Return a 2D array containing a Gabor wavelet.
    size -- (height, width) (pixels)
    alpha -- orientation (rad)
    phi -- phase (rad)
    freq -- frequency (cycles/deg)
    sgm -- (sigma_x, sigma_y) standard deviation along the axis
           of the gaussian ellipse (pixel)
    center -- (x,y) coordinates of the center of the wavelet (pixel)
            Default: None, meaning the center of the array
    res -- spatial resolution (deg/pixel)
            Default: 1, so that 'freq' is measured in cycles/pixel
    ampl -- constant multiplying the result
            Default: 1.
    """

    # Init
    w, h = size
    if center is None:
        center = (w // 2, h // 2)
    y0, x0 = center
    freq *= res
    sin_alpha = np.sin(alpha)
    cos_alpha = np.cos(alpha)
    v0, u0 = freq * cos_alpha, freq * sin_alpha

    # coordinates
    xy = np.meshgrid(np.arange(w) - x0, np.arange(h) - y0)
    xy = xy[0].T, xy[1].T
    xr = xy[0] * cos_alpha - xy[1] * sin_alpha
    yr = xy[0] * sin_alpha + xy[1] * cos_alpha

    return np.exp(-((xr/sgm[0])**2 + (yr/sgm[1])**2) / 2) * \
           np.sin(-2 * np.pi * (u0 * xy[0] + v0 * xy[1]) - phi) * \
           ampl


def gen_random_gabor_wts(shp):
    return gabor(shp,
                 alpha=2 * np.pi * runif(),
                 freq=runif(1, 5) / shp[int(runif() + .5)],
                 sgm=(runif(shp[0] / 5, shp[0] / 3),
                      runif(shp[1] / 5, shp[1] / 3)),
                 center=(runif(shp[0]), runif(shp[1])),
                 phi=2 * np.pi * runif())