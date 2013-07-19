import numpy as np
import healpytools as hpt
import coord
import _tpc


def twoPt(x, y, z, xt, yt, zt):
    """
    Angular two-point correlation helper function
    Takes two lists of normalized vectors (x,y,z) and (xt,yt,zt).
    Returns the minimum of angular distances in [rad] of each (x,y,z) to (xt,yt,zt)
    """
    dot = np.outer(x,xt) + np.outer(y,yt) + np.outer(z,zt)
    dot = np.clip(dot, -1., 1.) # catch numerical errors
    sep = np.arccos( dot )
    return sep.min(axis=1)

def twoPtAuto(x, y, z):
    """
    Angular two-point autocorrelation helper function
    Takes lists for x, y and z components of a number of vectors, these are assumed to be normalized to 1.
    Returns a list of angular distances in [rad] for every pair of vectors.
    """
    idx = np.triu_indices(len(x), 1) # indices of upper triangular without the diagonal
    dot = np.outer(x,x)[idx] + np.outer(y,y)[idx] + np.outer(z,z)[idx]
    dot = np.clip(dot, -1., 1.) # catch numerical errors
    return np.arccos(dot).flatten()

def tpac(x, y, z, w, maxangle, nbins, **kwargs):
    """ Angular two-point auto correlation. """
    ac = np.zeros(nbins)
    _tpc.tpac(ac, x, y, z, w, maxangle)
    
    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
            
    return ac

def tpcc(x1, y1, z1, w1, x2, y2, z2, w2, maxangle, nbins, **kwargs):
    """ Angular two-point cross correlation. """
    ac = np.zeros(nbins)
    _tpc.tpcc(ac, x1, y1, z1, w1, x2, y2, z2, w2, maxangle)
    
    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
            
    return ac


# def twoPtHealpix(m1, m2, bins):
#     """
#     Calculatetes the angular two-point correlation between 2 Healpix maps:
#     S(alpha) = sum_i sum_j 1 for alpha_ij < alpha
#     Bins are expected in [rad]
#     """
#     npix = len(m1)
#     nside = hpt.npix2nside(npix)
#     nbins = len(bins) - 1
#     x,y,z = hpt.pix2vec(nside, range(npix))
    
#     S = np.zeros(nbins)
#     for i in range(npix):
#         a = coord.angle(x[i], y[i], z[i], x[i:], y[i:], z[i:])
#         dig = np.digitize(a, bins)
#         s = np.bincount(dig, weights=m1[i] * m2[i:], minlength=nbins+2)
#         S += s[1:-1]

#     return np.cumsum(S)


if __name__ == "__main__":
    from pylab import *

    N = 1000
    phi = rand(N) * 2 * pi
    theta = arccos(rand(N) * 2 - 1)
    x = sin(theta) * cos(phi)
    y = sin(theta) * sin(phi)
    z = cos(theta)

    D = twoPtAuto(x,y,z) * 180/pi
    y, binedges = histogram(D, bins=100, range=(0,10))
    x = (binedges[1:] + binedges[:-1]) / 2
    y = cumsum(y)

    figure()
    plot(x, y)
    xlabel('Angular separation [deg]')
    ylabel('Number of pairs')
    show()
