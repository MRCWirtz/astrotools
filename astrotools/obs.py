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