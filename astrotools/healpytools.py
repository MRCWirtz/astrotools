# Extensions to healpy
import numpy as np
from healpy import *  # make healpy namespace available
import healpy
import coord


def randPixFromMap(map, n=1, nest=False):
    """
    Draw n random pixels from a HEALpix map.
    """
    p = np.cumsum(map)
    return p.searchsorted(np.random.rand(n) * p[-1])

def randVecInPix(nside, ipix, nest=False):
    """
    Draw vectors from a uniform distribution within a HEALpixel.
    nside : healpix nside parameter
    ipix  : pixel number(s)
    """
    if not(nest):
        ipix = healpy.ring2nest(nside, ipix=ipix)

    norder = nside2norder(nside)
    nUp = 29 - norder
    iUp = ipix * 4**nUp
    iUp += np.random.randint(0, 4**nUp, size=np.size(ipix))

    v = healpy.pix2vec(nside=2**29, ipix=iUp, nest=True)
    return np.array(v)

def randVecFromMap(map, n=1, nest=False):
    """
    Draw n random vectors from a HEALpix map.
    """
    pix = randPixFromMap(map, n, nest)
    nside = npix2nside(len(map))
    return randVecInPix(nside, pix, nest)

def pix2ang(i, nest=False):
    """
    Convert HEALpixel i to spherical angles (astrotools definition)
    Substitutes healpy.pix2ang
    """
    pass # not implemented

def ang2pix(phi, theta, nest=False):
    """
    Convert spherical angle (astrotools definition) to HEALpixel i
    Substitutes healpy.ang2pix
    """
    pass # not implemented

def angle(nside, i, j, nest=False):
    """
    Give the angular distance between two pixel.
    """
    v1 = pix2vec(nside, i, nest)
    v2 = pix2vec(nside, j, nest)
    return coord.angle(v1, v2)

def norder2npix(norder):
    """
    Give the number of pixel for the given HEALpix order.
    """
    return 12*4**norder

def npix2norder(npix):
    """
    Give the HEALpix order for the given number of pixel.
    """
    norder = np.log(npix/12) / np.log(4)
    if not(norder.is_integer()):
        raise ValueError('Wrong pixel number (it is not 12*4**norder)')
    return int(norder)

def norder2nside(norder):
    """
    Give the HEALpix nside parameter for the given HEALpix order.
    """
    return 2**norder

def nside2norder(nside):
    """
    Give the HEALpix order for the given HEALpix nside parameter.
    """
    norder = np.log2(nside)
    if not(norder.is_integer()):
        raise ValueError('Wrong nside number (it is not 2**norder)')
    return int(norder)



if __name__ == "__main__":
    import matplotlib.pyplot
    import coord

    # pixel 5 in base resolution (n = 0)
    norder = 0
    iPix = 5
    v = healpy.pix2vec(norder2nside(norder), iPix, nest=True)
    phi, theta = coord.vec2ang(*v)

    # centers of four-fold upsampled pixels
    nside_up = 2**(norder + 4)
    iPix_up = range(iPix * 4**4, (iPix+1) * 4**4)
    x, y, z = healpy.pix2vec(nside_up, iPix_up, nest=True)
    phi_up, theta_up = coord.vec2ang(x, y, z)

    # 20 random direction within the pixel
    v = randVecInPix(norder2nside(norder), np.ones(20, dtype=int) * iPix)
    phi_rnd, theta_rnd = coord.vec2ang(*v)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.plot(phi_up, theta_up, 'b+')
    ax.plot(phi_rnd, theta_rnd, 'go')
    ax.plot(phi, theta, 'ro')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
