import numpy as np
import healpy
import coord
import _tpc


def twoPtAuto(x, y, z, bins, **kwargs):
    """
    Angular two-point auto correlation.
    """
    idx = np.triu_indices(len(x), 1) # upper triangle indices without diagonal
    acu = coord.angle(x, y, z, x, y, z, each2each=True)[idx].flatten()
    dig = np.digitize(acu, bins)

    # optional weights
    w = kwargs.get('weights', None)
    if w != None:
        w = np.outer(w, w)[idx].flatten()

    ac = np.bincount(dig, minlength=len(bins)+1, weights=w)
    ac = ac.astype(float)[1:-1] # convert to float and remove overflow bins
    
    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
    if kwargs.get("normalized", False):
        if w != None:
            ac /= sum(w)
        else:
            ac /= (len(x)**2 - len(x)) / 2
    return ac

def twoPtCross(x1, y1, z1, x2, y2, z2, bins, **kwargs):
    """
    Angular two-point cross correlation.
    """
    ccu = coord.angle(x1, y1, z1, x2, y2, z2, each2each=True).flatten()
    dig = np.digitize(ccu, bins)

    # optional weights
    w1 = kwargs.get('weights1', None)
    w2 = kwargs.get('weights2', None)
    if (w1 != None) and (w2 != None):
        w = np.outer(w1, w2).flatten()
    else:
        w = None

    cc = np.bincount(dig, minlength=len(bins)+1, weights=w)
    cc = cc.astype(float)[1:-1]

    if kwargs.get("cumulative", True):
        cc = np.cumsum(cc)
    if kwargs.get("normalized", False):
        if w != None:
            cc /= sum(w)
        else:
            cc /= len(x1) * len(x2)
    return cc

def twoPtHealpix(m1, m2, maxangle, nbins):
    """
    Calculatetes the angular two-point correlation between 2 Healpix maps:
    S(alpha) = sum_i sum_j 1 for alpha_ij < alpha
    for angular bins in [rad]
    """
    npix = len(m1)
    nside = healpy.npix2nside(npix)
    x, y, z = healpy.pix2vec(nside, range(npix))
    i1 = (m1 != 0) # indices of non-zero elements
    i2 = (m2 != 0)

    cc = np.zeros(nbins)
    _tpc.wtpcc(cc, x[i1], y[i1], z[i1], m1[i1], x[i2], y[i2], z[i2], m2[i2], maxangle)
    cc = np.cumsum(cc)
    return cc

def tpac(x, y, z, maxangle, nbins, **kwargs):
    """
    Angular two-point auto correlation.
    """
    # optional weights
    w = kwargs.get('weights', np.ones(len(x)))
    
    ac = np.zeros(nbins)
    _tpc.wtpac(ac, x, y, z, w, maxangle)

    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
    if kwargs.get("normalized", False):
        n = len(x)
        ac /= (sum(w)**2 - sum(w**2)) / 2
    return ac

def tpcc(x1, y1, z1, x2, y2, z2, maxangle, nbins, **kwargs):
    """
    Angular two-point cross correlation with weights.
    """
    # optional weights
    w1 = kwargs.get('weights1', np.ones(len(x1)))
    w2 = kwargs.get('weights2', np.ones(len(x2)))

    cc = np.zeros(nbins)
    _tpc.wtpcc(cc, x1, y1, z1, w1, x2, y2, z2, w2, maxangle)

    if kwargs.get("cumulative", True):
        cc = np.cumsum(cc)
    if kwargs.get("normalized", False):
        cc /= sum(w1) * sum(w2)
    return cc