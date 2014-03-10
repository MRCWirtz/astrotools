import numpy as np


# Rotation matrix for the conversion : x_galactic = R * x_equatorial (J2000)
# http://adsabs.harvard.edu/abs/1989A&A...218..325M
_RGE = np.array([
    [-0.054875539, -0.873437105, -0.483834992],
    [+0.494109454, -0.444829594, +0.746982249],
    [-0.867666136, -0.198076390, +0.455983795]])

# Rotation matrix for the conversion : x_supergalactic = R * x_galactic
# http://iopscience.iop.org/0004-637X/530/2/625
_RSG = np.array([
    [-0.73574257480437488, +0.67726129641389432, 0.0000000000000000],
    [-0.07455377836523376, -0.08099147130697673, 0.9939225903997749],
    [+0.67314530210920764, +0.73127116581696450, 0.1100812622247821]])

# Rotation matrix for the conversion: x_equatorial = R * x_ecliptic
# http://en.wikipedia.org/wiki/Ecliptic_coordinate_system
_ecliptic = np.deg2rad(23.4)
_REE = np.array([
    [1.,                0.,                 0.],
    [0., np.cos(_ecliptic), -np.sin(_ecliptic)],
    [0., np.sin(_ecliptic),  np.cos(_ecliptic)]])


def eq2gal(x, y, z):
    """
    Rotate equatorial to galactical coordinates (same origin)
    """
    return _RGE.dot((x, y, z))


def gal2eq(x, y, z):
    """
    Rotate galactic to equatorial coordinates (same origin)
    """
    return _RGE.transpose().dot((x, y, z))


def gal2sgal(x, y, z):
    """
    Rotate galactic to supergalactic coordinates (same origin)
    """
    return _RSG.dot((x, y, z))


def sgal2gal(x, y, z):
    """
    Rotate supergalactic to galactic coordinates (same origin)
    """
    return _RSG.transpose().dot((x, y, z))


def ecl2eq(x, y, z):
    """
    Rotate ecliptic to equatorial coordinates (same origin)
    """
    return _REE.dot((x, y, z))


def eq2ecl(x, y, z):
    """
    Rotate equatorial to ecliptic coordinates (same origin)
    """
    return _REE.transpose().dot((x, y, z))


def dms2rad(degree, minutes, seconds):
    """
    Transform declination (degree, minute, second) to radians
    """
    s = -1. if degree < 0 else 1.
    return s * (np.fabs(degree) + 1. / 60 * minutes + 1. / 3600 * seconds) / 180. * np.pi


def hms2rad(hour, minutes, seconds):
    """
    Transform right ascension (hour, minute, second) to radians
    """
    return (hour / 12. + minutes / 720. + seconds / 43200.) * np.pi


def vec2ang(x, y, z):
    """
    Get spherical angles from vector
    (x, y, z) -> (phi, theta)
    phi (pi,-pi), 0 points in x-direction, pi/2 in y-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, (x * x + y * y)**.5)
    return (phi, theta)


def ang2vec(phi, theta):
    """
    Get vector from spherical angles
    (phi, theta) -> (x, y, z)
    phi (pi,-pi), 0 points in x-direction, pi/2 in y-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return (x, y, z)


def sphUnitVectors(phi, theta):
    """
    Get spherical unit vectors e_r, e_theta, e_phi from spherical angles
    """
    e_r = np.array([
        np.cos(theta) * np.cos(phi),
        np.cos(theta) * np.sin(phi),
        np.sin(theta)])
    e_theta = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        -np.cos(theta)])
    e_phi = np.array([-np.sin(phi), np.cos(phi), 0])
    return (e_r, e_theta, e_phi)


def distance(x1, y1, z1, x2, y2, z2):
    """
    Distance between each pair from two lists of vectors.
    """
    return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**.5


def angle(x1, y1, z1, x2, y2, z2, each2each=False):
    """
    Angular distance in [rad] for each pair from two lists of vectors.
    The vectors must be normalized.
    Use each2each=True to calculate every combination.
    """
    if each2each:
        d = np.outer(x1, x2) + np.outer(y1, y2) + np.outer(z1, z2)
    else:
        d = x1 * x2 + y1 * y2 + z1 * z2
    d = np.clip(d, -1., 1.)
    return np.arccos(d)


def minAngle(x1, y1, z1, x2, y2, z2):
    """
    Minimum angle of each vector (x1,y1,z1) to any of vectors (x2,y2,z2).
    """
    return angle(x1, y1, z1, x2, y2, z2, each2each=True).min(axis=1)


def exposureEquatorial(dec, a0=-35.25, zmax=60):
    """
    Relative exposure per solid angle of a detector
        at latitude a0 (-90, 90 degrees, default: Auger),
        with maximum acceptance zenith angle zmax (0, 90 degrees, default: 60)
        and for given equatorial declination dec (-pi/2, pi/2).
    See astro-ph/0004016
    """
    dec = np.array(dec)
    if (abs(dec) > np.pi/2).any():
        raise Exception('exposureEquatorial: declination not in range (-pi/2, pi/2)')
    if (zmax < 0) or (zmax > 90):
        raise Exception('exposureEquatorial: zmax not in range (0, 90 degrees)')
    if (a0 < -90) or (a0 > 90):
        raise Exception('exposureEquatorial: a0 not in range (-90, 90 degrees)')

    zmax *= np.pi/180
    a0 *= np.pi/180

    xi = (np.cos(zmax) - np.sin(a0) * np.sin(dec)) / np.cos(a0) / np.cos(dec)
    xi = np.clip(xi, -1, 1)
    am = np.arccos(xi)

    cov = np.cos(a0) * np.cos(dec) * np.sin(am) + am * np.sin(a0) * np.sin(dec)
    return cov / np.pi  # normalize the maximum possible value to 1

def randDec(n=1, a0=-35.25, zmax=60):
    """
    Returns n random equatorial declinations (pi/2, -pi/2) of a detector
        at latitude a0 (-90, 90 degrees, default: Auger),
        with maximum acceptance zenith angle zmax (0, 90 degrees, default: 60)
    See coord.exposureEquatorial
    """
    # sample probability distribution using the rejection technique
    nTry = int(3.3 * n) + 50
    dec = np.arcsin( 2*np.random.rand(nTry) - 1 )
    maxVal = 0.58  # FIXME: this works for Auger declination and zmax only
    accept = coord.exposureEquatorial(dec, a0, zmax) > np.random.rand(nTry) * maxVal
    if sum(accept) < n:
        raise Exception("randEqDec: stochastic failure")
    return dec[accept][:n]

def randPhi(n=1):
    """
    Random uniform phi (-pi, pi).
    """
    return (np.random.rand(n) * 2 - 1) * np.pi


def randTheta(n=1):
    """
    Random theta (pi/2, -pi/2) from uniform cos(theta) distribution.
    """
    return np.pi/2 - np.arccos(np.random.rand(n) * 2 - 1)
