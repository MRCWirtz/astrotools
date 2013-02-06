import math


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
    x = r * math.cos(theta) * math.cos(phi)
    y = r * math.cos(theta) * math.sin(phi)
    z = r * math.sin(theta)
    return (x, y, z)

def cartesian2Spherical(x, y, z):
    """
    Transform cartesian to spherical coordinates
    (x, y, z) -> (r, phi, theta)
    phi (pi,-pi), 0 points in x-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    r = (x * x + y * y + z * z)**.5
    phi = math.atan2(y, x)
    theta = math.atan2(z, (x * x + y * y)**.5)
    return (r, phi, theta)

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

