import numpy as np
import coord


def twoPtAuto(v, bins=np.arange(0, 181, 1), **kwargs):
    """
    Angular two-point auto correlation for a set of directions v.
    WARNING: Due to the vectorized calculation this function
    does not work for large numbers of events.

    Parameters
    ----------
    v : directions, (3 x n) matrix with the rows holding x,y,z
    bins : angular bins in degrees
    weights : weights for each event (optional)
    cumulative : make cumulative (default=True)
    normalized : normalize to 1 (default=False)
    """
    n = np.shape(v)[1]
    idx = np.triu_indices(n, 1)  # upper triangle indices without diagonal
    ang = coord.angle(v, v, each2each=True)[idx]

    # optional weights
    w = kwargs.get('weights', None)
    if w is not None:
        w = np.outer(w, w)[idx]

    dig = np.digitize(ang, bins)
    ac = np.bincount(dig, minlength=len(bins)+1, weights=w)
    ac = ac.astype(float)[1:-1]  # convert to float and remove overflow bins

    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
    if kwargs.get("normalized", False):
        if w is not None:
            ac /= sum(w)
        else:
            ac /= (n**2 - n) / 2
    return ac


def twoPtCross(v1, v2, bins=np.arange(0, 181, 1), **kwargs):
    """
    Angular two-point cross correlation for two sets of directions v1, v2.

    Parameters
    ----------
    v1: directions, (3 x n1) matrix with the rows holding x,y,z
    v2: directions, (3 x n2) matrix with the rows holding x,y,z
    bins : angular bins in degrees
    weights1, weights2 : weights for each event (optional)
    cumulative : make cumulative (default=True)
    normalized : normalize to 1 (default=False)
    """
    ang = coord.angle(v1, v2, each2each=True).flatten()
    dig = np.digitize(ang, bins)

    # optional weights
    w1 = kwargs.get('weights1', None)
    w2 = kwargs.get('weights2', None)
    if (w1 is not None) and (w2 is not None):
        w = np.outer(w1, w2).flatten()
    else:
        w = None

    cc = np.bincount(dig, minlength=len(bins)+1, weights=w)
    cc = cc.astype(float)[1:-1]

    if kwargs.get("cumulative", True):
        cc = np.cumsum(cc)
    if kwargs.get("normalized", False):
        if w is not None:
            cc /= sum(w)
        else:
            n1 = np.shape(v1)[1]
            n2 = np.shape(v2)[1]
            cc /= n1 * n2
    return cc


def thrust(P, ntry=5000):
    """
    Thrust observable for an array (n x 3) of 3-momenta.
    Returns 3 values (thrust, thrust major, thrust minor)
    and the 3 corresponding axes.
    """
    # thrust
    n1 = np.sum(P, axis=0)
    n1 /= np.linalg.norm(n1)
    t1 = np.sum(abs(np.dot(P, n1.T)), axis=0)

    # thrust major, brute force calculation
    er, et, ep = coord.sphUnitVectors(*coord.vec2ang(*n1))
    alpha = np.linspace(0, np.pi, ntry)
    n2try = np.outer(np.cos(alpha), et) + np.outer(np.sin(alpha), ep)
    t2try = np.sum(abs(np.dot(P, n2try.T)), axis=0)
    i = np.argmax(t2try)
    n2 = n2try[i]
    t2 = t2try[i]

    # thrust minor
    n3 = np.cross(n1, n2)
    t3 = np.sum(abs(np.dot(P, n3.T)), axis=0)

    # normalize
    sumP = np.sum(np.sum(P**2, axis=1)**.5)
    T = np.array((t1, t2, t3)) / sumP
    N = np.array((n1, n2, n3))
    return T, N
