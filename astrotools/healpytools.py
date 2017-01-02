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

def pix2ang(nside, ipix, nest=False):
    """
    Convert HEALpixel ipix to spherical angles (astrotools definition)
    Substitutes healpy.pix2ang
    """
    v = healpy.pix2vec(nside, ipix)
    phi, theta = coord.vec2ang(v)

    return (phi, theta)

def ang2pix(nside, phi, theta, nest=False):
    """
    Convert spherical angle (astrotools definition) to HEALpixel ipix
    Substitutes healpy.ang2pix
    """
    v = coord.ang2vec(phi, theta)
    ipix = healpy.vec2pix(nside, *v)

    return ipix

def angle(nside, ipix, jpix, nest=False):
    """
    Give the angular distance between two pixel.
    """
    v1 = pix2vec(nside, ipix, nest)
    v2 = pix2vec(nside, jpix, nest)
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


def statistic(nside, x, y, z, statistic='count', vals=None):
    """
    Create HEALpix map of count, frequency or mean or rms value.

    Parameters
    ----------
    nside: int
        Healpix nside parameter = 4^norder, norder = 0, 1, ..
        Lenses use nside = 64 (norder = 6)
    x, y, z: array_like
        coordinates
    statistic: keyword
        'count', 'frequency', 'mean' or 'rms'
    vals: array_like
        values for which the mean or rms is calculated
    """
    npix = nside2npix(nside)
    pix = vec2pix(nside, x, y, z)
    nmap = np.bincount(pix, minlength=npix)

    if statistic == 'count':
        vmap = nmap.astype('float')

    elif statistic == 'frequency':
        vmap = nmap.astype('float')
        vmap /= max(nmap)# frequency [0,1]

    elif statistic == 'mean':
        if vals == None: raise ValueError
        vmap = np.bincount(pix, weights=vals, minlength=npix)
        vmap /= nmap # mean

    elif statistic == 'rms':
        if vals == None: raise ValueError
        vmap = np.bincount(pix, weights=vals**2, minlength=npix)
        vmap = (vmap / nmap)**.5 # rms

    vmap[nmap==0] = UNSEEN
    return vmap


def fisher_pdf(nside, x, y, z, k, threshold=4):
    """
    Fisher distribution of healpy pixels with mean direction (x, y, z) and concentration parameter
    kappa normalized to 1

    :param nside: nside of the healpy map
    :param x: x-coordinate of the center
    :param y: y-coordinate of the center
    :param z: z-coordinate of the center
    :param k: kappa for the fisher distribution, 1 / sigma**2
    :param threshold: Threshold in sigma up to where the distribution should be calculated
    :return: pixels, values at pixels
    """
    length = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    sigma = 1. / np.sqrt(k)  # in radians
    # if alpha_max is larger than a reasonable np.pi than query disk takes care of using only
    # np.pi as maximum range.
    alpha_max = threshold * sigma

    pixels = healpy.query_disc(nside, (x, y, z), alpha_max)
    px, py, pz = healpy.pix2vec(nside, pixels)
    d = (x * px + y * py + z * pz) / length
    # for large values of kappa exp(k * d) goes to infinity which is meaningless. So we use the trick to write:
    # exp(k * d) = exp(k * d + k - k) = exp(k) * exp(k * (d-1)). As we normalize the function to one in the end, we can
    # leave out the first factor exp(k)
    weights = np.exp(k * (d - 1)) if k > 30 else np.exp(k * d)
    return pixels, weights / np.sum(weights)
    # if you want to normalize on the number of pixels use
    # norm = k / (np.exp(k) - np.exp(-k)) / 2 / np.pi if k < 30 else k / np.exp(k) / 2 / np.pi
    # weights = norm * np.exp(k * d)
    # return pixels, weights


def dipole_pdf(nside, a, x, y=None, z=None):
    """
    Probability density function of a dipole. Returns 1 + a * cos(theta) for all pixels in
    hp.nside2npix(nside)

    :param nside: nside of the healpy map
    :param a: amplitude of the dipole, 0 <= a <= 1, automatically clipped
    :param x: x-coordinate of the center or numpy array with center coordinates
    :param y: y-coordinate of the center
    :param z: z-coordinate of the center
    :return: weights
    """
    a = np.clip(a, 0., 1.)
    direction = np.array(x, dtype=np.float) if (y is None and z is None) else np.array([x, y, z],
        dtype=np.float)
    # normalize to one
    direction /= np.sqrt(np.sum(direction**2))
    npix = healpy.nside2npix(nside)
    v = np.array(healpy.pix2vec(nside, np.arange(npix)))
    cosangle = np.sum(v.T * direction, axis=1)
    return 1 + a * cosangle


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
