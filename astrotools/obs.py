import numpy as np
import coord


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

def thrust(P, ntry=5000):
    """
    Thrust observable for an array (n x 3) of 3-momenta.
    Returns 3 values (thrust, thrust major, thrust minor) and the 3 corresponding axes.
    """
    # thrust
    n1 = sum(P, axis=0)
    n1 /= norm(n1)
    t1 = sum(abs(dot(P, n1.T)), axis=0)
    
    # thrust major, brute force calculation
    er, et, ep = coord.sphUnitVectors(*coord.vec2ang(*n1))
    alpha = linspace(0, pi, ntry)
    n2try = outer(cos(alpha), et) + outer(sin(alpha), ep)
    t2try = sum(abs(dot(P, n2try.T)), axis=0)
    i = argmax(t2try)
    n2 = n2try[i]
    t2 = t2try[i]

    # thrust minor
    n3 = cross(n1, n2)
    t3 = sum(abs(dot(P, n3.T)), axis=0)

    # normalize
    sumP = sum(sum(P**2, axis=1)**.5)
    T = array((t1, t2, t3)) / sumP
    V = array((n1, n2, n3))
    return T, V