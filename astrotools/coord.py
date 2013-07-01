import numpy as np


# Rotation matrix for conversion : x_galactic = R * x_equatorial (J2000)
# http://adsabs.harvard.edu/abs/1989A&A...218..325M
_R2000 = [
    [-0.054875539, -0.873437105, -0.483834992],
    [ 0.494109454, -0.444829594,  0.746982249],
    [-0.867666136, -0.198076390,  0.455983795]]

# Rotation matrix for conversion : x_supergalactic = R * x_galactic
# http://iopscience.iop.org/0004-637X/530/2/625
_RSG = [
    [-0.73574257480437488,  0.67726129641389432, 0.0000000000000000],
    [-0.07455377836523376, -0.08099147130697673, 0.9939225903997749],
    [ 0.67314530210920764,  0.73127116581696450, 0.1100812622247821]]

def spherical2Cartesian(r, phi, theta):
    """
    Transform spherical to cartesian coordinates
    (r, phi, theta) -> (x, y, z)
    phi (pi,-pi), 0 points in x-direction, pi/2 in y-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    x = r * np.cos(theta) * np.cos(phi)
    y = r * np.cos(theta) * np.sin(phi)
    z = r * np.sin(theta)
    return (x, y, z)

def cartesian2Spherical(x, y, z):
    """
    Transform cartesian to spherical coordinates
    (x, y, z) -> (r, phi, theta)
    phi (pi,-pi), 0 points in x-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    r = (x * x + y * y + z * z)**.5
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, (x * x + y * y)**.5)
    return (r, phi, theta)

def vec2Ang(x, y, z):
    """
    Get spherical angles from vector
    Angle definitions correspond to cartesian2Spherical
    """
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, (x * x + y * y)**.5)
    return (phi, theta)

def ang2Vec(phi, theta):
    """
    Get vector from spherical angles
    Angle definitions correspond to spherical2Cartesian
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return (x, y, z)

def randPhi(n=1):
    """
    Random uniform phi (-pi, pi).
    """
    return (np.random.rand(n) * 2 - 1) * np.pi

def randTheta(n=1):
    """
    Random theta (pi/2, -pi/2) from uniform cos(theta) distribution.
    """
    return np.pi/2 - np.arccos( np.random.rand(n) * 2 - 1 )

def distance(x1, y1, z1, x2, y2, z2):
    """
    Distance between 2 vectors (= 2-norm of connecting vector).
    """
    return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**.5

def angle(x1, y1, z1, x2, y2, z2):
    """
    Angular separation in rad between two given normalized vectors.
    """
    d = x1 * x2 + y1 * y2 + z1 * z2
    d = np.clip(d, -1., 1.)
    return np.arccos(d)

def galactic2Equatorial(x, y, z):
    """
    Rotate galactic to equatorial coordinates (same origin)
    """
    _x = _R2000[0][0] * x + _R2000[1][0] * y + _R2000[2][0] * z
    _y = _R2000[0][1] * x + _R2000[1][1] * y + _R2000[2][1] * z
    _z = _R2000[0][2] * x + _R2000[1][2] * y + _R2000[2][2] * z

    return (_x, _y, _z)

def equatorial2Galactic(x, y, z):
    """
    Rotate equatorial to galactical coordinates (same origin)
    """
    _x = _R2000[0][0] * x + _R2000[0][1] * y + _R2000[0][2] * z
    _y = _R2000[1][0] * x + _R2000[1][1] * y + _R2000[1][2] * z
    _z = _R2000[2][0] * x + _R2000[2][1] * y + _R2000[2][2] * z
    return (_x, _y, _z)

def galactic2Supergalactic(x, y, z):
    """
    Rotate galactic to supergalactic coordinates (same origin)
    """
    _x = _RSG[0][0] * x + _RSG[0][1] * y + _RSG[0][2] * z
    _y = _RSG[1][0] * x + _RSG[1][1] * y + _RSG[1][2] * z
    _z = _RSG[2][0] * x + _RSG[2][1] * y + _RSG[2][2] * z
    return (_x, _y, _z)

def supergalactic2Galactic(x, y, z):
    """
    Rotate supergalactic to galactic coordinates (same origin)
    """
    _x = _RSG[0][0] * x + _RSG[1][0] * y + _RSG[2][0] * z
    _y = _RSG[0][1] * x + _RSG[1][1] * y + _RSG[2][1] * z
    _z = _RSG[0][2] * x + _RSG[1][2] * y + _RSG[2][2] * z
    return (_x, _y, _z)

def supergalactic2GalacticSpherical(r, phi, theta):
    """
    Transform supergalactic to galactic spherical coordinates.
    """
    sgx, sgy, sgz = spherical2Cartesian(r, phi, theta)
    gx, gy, gz = supergalactic2Galactic(sgx, sgy, sgz)
    return cartesian2Spherical(gx, gy, gz)

def galactic2SupergalacticSpherical(r, phi, theta):
    """
    Transform galactic to supergalactic spherical coordinates.
    """
    sgx, sgy, sgz = spherical2Cartesian(r, phi, theta)
    gx, gy, gz = galactic2Supergalactic(sgx, sgy, sgz)
    return cartesian2Spherical(gx, gy, gz)

def dms2rad(degree, minutes, seconds):
	"""
	Transform declination (degree, minute, second) to radians
	"""
	s = -1. if degree < 0 else 1.
	return s * (math.fabs(degree) + 1. / 60 * minutes + 1. / 3600 * seconds)	/ 180. * math.pi;

def hms2rad(hour, minutes, seconds):
	"""
	Transform right ascension (hour, minute, second) to radians
	"""
	return (hour / 12. + minutes / 720. + seconds / 43200.) * math.pi;
