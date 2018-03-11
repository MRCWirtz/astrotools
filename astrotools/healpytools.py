"""
Extensions to healpy that covers:
-> pixel, vector, angular transformations
-> drawing vectors uniformly in pixel
-> various probability density functions (exposure, fisher, dipole)
"""

import healpy as hp
# noinspection PyUnresolvedReferences
from healpy import *
import numpy as np

from astrotools import coord


def rand_pix_from_map(healpy_map, n=1):
    """
    Draw n random pixels from a HEALpix map.

    :param healpy_map: healpix map (not necessarily normalized)
    :param n: number of pixels that are drawn from the map
    :return: an array of pixels with size n, that are drawn from the map
    """
    p = np.cumsum(healpy_map)
    return p.searchsorted(np.random.rand(n) * p[-1])


def rand_vec_in_pix(nside, ipix, nest=False):
    """
    Draw vectors from a uniform distribution within a HEALpixel.

    :param nside: nside of the healpy pixelization
    :param ipix: pixel number(s)
    :param nest: set True in case you work with healpy's nested scheme
    :return: vectors containing events from the pixel(s) specified in ipix
    """
    if not nest:
        ipix = hp.ring2nest(nside, ipix=ipix)

    n_order = nside2norder(nside)
    n_up = 29 - n_order
    i_up = ipix * 4 ** n_up
    i_up += np.random.randint(0, 4 ** n_up, size=np.size(ipix))

    v = hp.pix2vec(nside=2 ** 29, ipix=i_up, nest=True)
    return np.array(v)


def rand_vec_from_map(healpy_map, n=1, nest=False):
    """
    Draw n random vectors from a HEALpix map.

    :param healpy_map: healpix map (not necessarily normalized)
    :param n: number of pixels that are drawn from the map
    :param nest: set True in case you work with healpy's nested scheme
    :return: an array of vectors with size n, that are drawn from the map
    """
    pix = rand_pix_from_map(healpy_map, n)
    nside = hp.npix2nside(len(healpy_map))
    return rand_vec_in_pix(nside, pix, nest)


def pix2ang(nside, ipix, nest=False):
    """
    Convert HEALpixel ipix to spherical angles (astrotools definition)
    Substitutes hp.pix2ang

    :param nside: nside of the healpy pixelization
    :param ipix: pixel number(s)
    :param nest: set True in case you work with healpy's nested scheme
    :return: angles (phi, theta) in astrotools definition
    """
    v = hp.pix2vec(nside, ipix, nest=nest)
    phi, theta = coord.vec2ang(v)

    return phi, theta


def pix2vec(nside, ipix, nest=False):
    """
    Convert HEALpixel ipix to cartesian vector
    Substitutes hp.pix2vec

    :param nside: nside of the healpy pixelization
    :param ipix: pixel number(s)
    :param nest: set True in case you work with healpy's nested scheme
    :return: vector of the pixel center(s)
    """
    v = hp.pix2vec(nside, ipix, nest=nest)
    return v


def ang2pix(nside, phi, theta, nest=False):
    """
    Convert spherical angle (astrotools definition) to HEALpixel ipix
    Substitutes hp.ang2pix

    :param nside: nside of the healpy pixelization
    :param phi: longitude in astrotools definition
    :param theta: latitude in astrotools definition
    :param nest: set True in case you work with healpy's nested scheme
    :return: pixel number(s)
    """
    v = coord.ang2vec(phi, theta)
    ipix = hp.vec2pix(nside, *v, nest=nest)

    return ipix


def vec2pix(nside, x, y, z, nest=False):
    """
    Convert HEALpixel ipix to spherical angles (astrotools definition)
    Substitutes hp.vec2pix

    :param nside: nside of the healpy pixelization
    :param x: x-coordinate of the center
    :param y: y-coordinate of the center
    :param z: z-coordinate of the center
    :param nest: set True in case you work with healpy's nested scheme
    :return: vector of the pixel center(s)
    """
    ipix = hp.vec2pix(nside, x, y, z, nest=nest)
    return ipix


def angle(nside, ipix, jpix, nest=False):
    """
    Give the angular distance between two pixel.

    :param nside: nside of the healpy pixelization
    :param ipix: healpy pixel i (either int or array like int)
    :param jpix: healpy pixel j (either int or array like int)
    :param nest: use the nesting scheme of healpy
    """
    v1 = hp.pix2vec(nside, ipix, nest)
    v2 = hp.pix2vec(nside, jpix, nest)
    return coord.angle(v1, v2)


def norder2npix(norder):
    """
    Give the number of pixel for the given HEALpix order.

    :param norder: norder of the healpy pixelization
    :return: npix: number of pixels of the healpy pixelization
    """
    return 12 * 4 ** norder


def npix2norder(npix):
    """
    Give the HEALpix order for the given number of pixel.

    :param npix: number of pixels of the healpy pixelization
    :return: norder: norder of the healpy pixelization
    """
    norder = np.log(npix / 12) / np.log(4)
    if not (norder.is_integer()):
        raise ValueError('Wrong pixel number (it is not 12*4**norder)')
    return int(norder)


def norder2nside(norder):
    """
    Give the HEALpix nside parameter for the given HEALpix order.

    :param norder: norder of the healpy pixelization
    :return: nside: nside of the healpy pixelization
    """
    return 2 ** norder


def nside2norder(nside):
    """
    Give the HEALpix order for the given HEALpix nside parameter.

    :param nside: nside of the healpy pixelization
    :return: norder: norder of the healpy pixelization
    """
    norder = np.log2(nside)
    if not (norder.is_integer()):
        raise ValueError('Wrong nside number (it is not 2**norder)')
    return int(norder)


def statistic(nside, x, y, z, statistics='count', vals=None):
    """
    Create HEALpix map of count, frequency or mean or rms value.

    :param nside: nside of the healpy pixelization
    :param x: x-coordinate of the center
    :param y: y-coordinate of the center
    :param z: z-coordinate of the center
    :param statistics: keywords 'count', 'frequency', 'mean' or 'rms' possible
    :param vals: values (array like) for which the mean or rms is calculated
    :return: either count, frequency, mean or rms maps
    """
    npix = hp.nside2npix(nside)
    pix = hp.vec2pix(nside, x, y, z)
    n_map = np.bincount(pix, minlength=npix)

    if statistics == 'count':
        v_map = n_map.astype('float')

    elif statistics == 'frequency':
        v_map = n_map.astype('float')
        v_map /= max(n_map)  # frequency [0,1]

    elif statistics == 'mean':
        if vals is None:
            raise ValueError
        v_map = np.bincount(pix, weights=vals, minlength=npix)
        v_map /= n_map  # mean

    elif statistics == 'rms':
        if vals is None:
            raise ValueError
        v_map = np.bincount(pix, weights=vals ** 2, minlength=npix)
        v_map = (v_map / n_map) ** .5  # rms

    # noinspection PyUnboundLocalVariable
    v_map[n_map == 0] = hp.UNSEEN
    return v_map


def skymap_mean_quantile(skymap, quantile=0.68):
    """
    Calculates mean direction and quantile from an arbitrary healpy skymap

    :param skymap: healpy skymap of valid healpy size
    :param quantile: quantile for which the scatter angle is calculated
    :return: either count, frequency, mean or rms maps
    """
    npix = len(skymap)
    nside = hp.npix2nside(npix)
    norm = np.sum(skymap)
    skymap /= norm

    if norm == 0:
        raise Exception("Given skymap is empty.")

    # calculate mean vector
    vecs = pix2vec(nside, range(npix))
    vec_mean = np.sum(vecs * skymap[np.newaxis], axis=1)
    vec_mean /= np.sqrt(np.sum(vec_mean**2))

    # calculate alpha
    alpha = np.arccos(np.sum(vecs * vec_mean[:, np.newaxis], axis=0))
    srt = np.argsort(alpha)
    idx = np.searchsorted(np.cumsum(skymap[srt]), quantile)
    alpha_quantile = alpha[srt][idx]

    return vec_mean, alpha_quantile


def exposure_pdf(nside=64, a0=-35.25, zmax=60):
    """
    Exposure (type: density function) of an experiment located at equatorial
    declination a0 and measuring events with zenith angles up to zmax degrees.

    :param nside: nside of the output healpy map
    :param a0: equatorial declination [deg] of the experiment (default: AUGER, a0=-35.25 deg)
    :param zmax: maximum zenith angle [deg] for the events
    :return: weights of the exposure map
    """
    npix = hp.nside2npix(nside)
    v_gal = pix2vec(nside, range(npix))
    v_eq = coord.gal2eq(v_gal)
    _, theta = coord.vec2ang(v_eq)
    exposure = coord.exposure_equatorial(theta, a0, zmax)
    exposure /= exposure.sum()
    return exposure


def fisher_pdf(nside, x, y, z, k, threshold=4, sparse=False):
    """
    Probability density function of a fisher distribution of healpy pixels with mean direction (x, y, z) and
    concentration parameter kappa; normalized to 1.

    :param nside: nside of the healpy map
    :param x: x-coordinate of the center
    :param y: y-coordinate of the center
    :param z: z-coordinate of the center
    :param k: kappa for the fisher distribution, 1 / sigma**2
    :param threshold: Threshold in sigma up to where the distribution should be calculated
    :param sparse: returns the map in the form (pixels, weights); this may be meaningfull for small distributions
    :return: pixels, weights at pixels
    """
    length = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    sigma = 1. / np.sqrt(k)  # in radians
    # if alpha_max is larger than a reasonable np.pi than query disk takes care of using only
    # np.pi as maximum range.
    alpha_max = threshold * sigma

    pixels = np.array(hp.query_disc(nside, (x, y, z), alpha_max))
    if len(pixels) == 0:
        # If sigma is too small, the pixel sequence will be empty
        pixels = np.array([vec2pix(nside, x, y, z)])
        weights = np.array([1.])
    else:
        px, py, pz = hp.pix2vec(nside, pixels)
        d = (x * px + y * py + z * pz) / length
        # for large values of kappa exp(k * d) goes to infinity which is meaningless. So we use the trick to write:
        # exp(k * d) = exp(k * d + k - k) = exp(k) * exp(k * (d-1)). As we normalize the function to one in the end,
        # we can leave out the first factor exp(k)
        weights = np.exp(k * (d - 1)) if k > 30 else np.exp(k * d)
        weights /= np.sum(weights)

    if sparse:
        return pixels, weights

    fisher_map = np.zeros(hp.nside2npix(nside))
    fisher_map[pixels] = weights
    return fisher_map


def dipole_pdf(nside, a, x, y=None, z=None):
    """
    Probability density function of a dipole. Returns 1 + a * cos(theta) for all pixels in hp.nside2npix(nside).

    :param nside: nside of the healpy map
    :param a: amplitude of the dipole, 0 <= a <= 1, automatically clipped
    :param x: x-coordinate of the center or numpy array with center coordinates (cartesian definition)
    :param y: y-coordinate of the center
    :param z: z-coordinate of the center
    :return: weights
    """
    a = np.clip(a, 0., 1.)
    if y is None and z is None:
        direction = np.array(x, dtype=np.float)
    else:
        direction = np.array([x, y, z], dtype=np.float)

    # normalize to one
    direction /= np.sqrt(np.sum(direction ** 2))
    npix = hp.nside2npix(nside)
    v = np.array(hp.pix2vec(nside, np.arange(npix)))
    cos_angle = np.sum(v.T * direction, axis=1)

    return 1 + a * cos_angle
