"""
Tools for coordinate transformation and vector calculations
"""
import numpy as np

# Rotation matrix for the conversion : x_galactic = R * x_equatorial (J2000)
# http://adsabs.harvard.edu/abs/1989A&A...218..325M
_RGE = np.array([[-0.054875539, -0.873437105, -0.483834992],
                 [+0.494109454, -0.444829594, +0.746982249],
                 [-0.867666136, -0.198076390, +0.455983795]])

# Rotation matrix for the conversion : x_supergalactic = R * x_galactic
# http://iopscience.iop.org/0004-637X/530/2/625
_RSG = np.array([[-0.73574257480437488, +0.67726129641389432, 0.0000000000000000],
                 [-0.07455377836523376, -0.08099147130697673, 0.9939225903997749],
                 [+0.67314530210920764, +0.73127116581696450, 0.1100812622247821]])

# Rotation matrix for the conversion: x_equatorial = R * x_ecliptic
# http://en.wikipedia.org/wiki/Ecliptic_coordinate_system
ECLIPTIC = np.deg2rad(23.44)
_REE = np.array([[1., 0., 0.],
                 [0., np.cos(ECLIPTIC), -np.sin(ECLIPTIC)],
                 [0., np.sin(ECLIPTIC), np.cos(ECLIPTIC)]])


def eq2gal(v):
    """
    Rotate equatorial to galactical coordinates (same origin)

    :param v: cartesian vector(s) of shape (3, n) in equatorial coordinate system
    :return: vector(s) of shape (3, n) in galactic coordinate system
    """
    return np.dot(_RGE, np.asarray(v))


def gal2eq(v):
    """
    Rotate galactic to equatorial coordinates (same origin)

    :param v: cartesian vector(s) of shape (3, n) in galactic coordinate system
    :return: vector(s) of shape (3, n) in equatorial coordinate system
    """
    return np.dot(_RGE.transpose(), np.asarray(v))


def gal2sgal(v):
    """
    Rotate galactic to supergalactic coordinates (same origin)

    :param v: cartesian vector(s) of shape (3, n) in galactic coordinate system
    :return: vector(s) of shape (3, n) in supergalactic coordinate system
    """
    return np.dot(_RSG, np.asarray(v))


def sgal2gal(v):
    """
    Rotate supergalactic to galactic coordinates (same origin)

    :param v: cartesian vector(s) of shape (3, n) in supergalactic coordinate system
    :return: vector(s) of shape (3, n) in galactic coordinate system
    """
    return np.dot(_RSG.transpose(), np.asarray(v))


def ecl2eq(v):
    """
    Rotate ecliptic to equatorial coordinates (same origin)

    :param v: cartesian vector(s) of shape (3, n) in ecliptic coordinate system
    :return: vector(s) of shape (3, n) in equatorial coordinate system
    """
    return np.dot(_REE, np.asarray(v))


def eq2ecl(v):
    """
    Rotate equatorial to ecliptic coordinates (same origin)

    :param v: cartesian vector(s) of shape (3, n) in equatorial coordinate system
    :return: vector(s) of shape (3, n) in ecliptic coordinate system
    """
    return np.dot(_REE.transpose(), np.asarray(v))


def dms2rad(degree, minutes, seconds):
    """
    Transform declination (degree, minute, second) to radians

    :param degree: integer angle of declination in degree
    :param minutes: arc minute of the declination angle
    :param seconds: arc second of the declination angle
    :return: declination angle in radians
    """
    return np.sign(degree) * (np.fabs(degree) + 1. / 60 * minutes + 1. / 3600 * seconds) / 180. * np.pi


def hms2rad(hour, minutes, seconds):
    """
    Transform right ascension (hour, minute, second) to radians

    :param hour: integer hour of right ascension
    :param minutes: arc minute of the right ascension angle
    :param seconds: arc second of the right ascension angle
    :return: right ascension angle in radians
    """
    return (hour / 12. + minutes / 720. + seconds / 43200.) * np.pi


def get_hour_angle(ra, lst):
    """
    Returns the hour angle (in radians) for a specific right ascension and local sidereal time

    :param ra: right ascension in radians
    :param lst: local sidereal time in radians
    :return: hour angle in radians
    """
    return (lst - ra) % (2 * np.pi)


def alt2zen(elevation):
    """
    Transforms an elevation angle [radians] in zenith angles
    :param elevation: elevation angle in radians
    :return: zenith angle in degrees
    """
    return np.pi / 2. - elevation


def date_to_julian_day(year, month, day):
    """
    Returns the Julian day number of a date from:
    http://code-highlights.blogspot.de/2013/01/julian-date-in-python.html
    http://code.activestate.com/recipes/117215/
    """
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def get_greenwich_siderial_time(time):
    """
    Convert civil time to (mean, wrt the mean equinox) Greenwich sidereal time.
    uncertainty of not taking the apparent time (wrt true equinox) is less then 0.01 deg
    time must be a datetime object
    adapted from http://infohost.nmt.edu/tcc/help/lang/python/examples/sidereal/ims/SiderealTime-gst.html

    :param time: class instance of datetime.date
    :return gst: Greenwich sidereal time
    """
    import datetime
    # [ nDays  :=  number of days between January 0.0 and utc ]
    date_ord = time.toordinal()
    jan1_ord = datetime.date(time.year, 1, 1).toordinal()
    n_days = date_ord - jan1_ord + 1

    jan_dt = datetime.datetime(time.year, 1, 1)
    jan_jd = date_to_julian_day(jan_dt.year, jan_dt.month, jan_dt.day) - 1.5
    s = jan_jd - 2415020.0
    t = s / 36525.0
    r = (0.00002581 * t + 2400.051262) * t + 6.6460656
    u = r - 24 * (time.year - 1900)
    factor_b = 24.0 - u

    sidereal_a = 0.0657098
    t0 = (n_days * sidereal_a - factor_b)
    dec_utc = time.hour + 1. / 60. * time.minute + 1. / 3600. * (time.second + time.microsecond * 1.e-6)
    sidereal_c = 1.002738
    gst = (dec_utc * sidereal_c + t0) % 24.0
    return gst


def get_local_sidereal_time(time, longitude):
    """
    Convert civil time to local sidereal time

    :param time: class instance of datetime.date
    :return lst: Local sidereal time (in radians)
    """
    gst = get_greenwich_siderial_time(time)
    gst *= np.pi / 12.
    return (gst + longitude) % (2 * np.pi)


def normed(v):
    """
    Return the normalized (lists of) vectors.

    :param v: vector(s) of shape (3, n)
    :return: corresponding normalized vectors of shape (3, n)
    """
    return np.asarray(v) / np.linalg.norm(v, axis=0)


def distance(v1, v2):
    """
    Linear distance between each pair from two (lists of) vectors.

    :param v1: vector(s) of shape (3, n)
    :param v2: vector(s) of shape (3, n)
    :return: linear distance between each pair
    """
    return np.linalg.norm(np.asarray(v1) - np.asarray(v2), axis=0)


def angle(v1, v2, each2each=False):
    """
    Angular distance in radians for each pair from two (lists of) vectors.
    Use each2each=True to calculate every combination.

    :param v1: vector(s) of shape (3, n)
    :param v2: vector(s) of shape (3, n)
    :param each2each: if true calculates every combination of the two lists v1, v2
    :return: angular distance in radians
    """
    a = normed(v1)
    b = normed(v2)
    if each2each:
        d = np.outer(a[0], b[0]) + np.outer(a[1], b[1]) + np.outer(a[2], b[2])
    else:
        if len(a.shape) == 1:
            a = a.reshape(3, 1)
        if len(b.shape) == 1:
            b = b.reshape(3, 1)
        d = np.sum(a * b, axis=0)
    return np.arccos(np.clip(d, -1., 1.))


def vec2ang(v):
    """
    Get spherical angles (phi, theta) from a (list of) vector(s).

    :param v: vector(s) of shape (3, n)
    :return: tuple consisting of

             - phi [range (pi, -pi), 0 points in x-direction, pi/2 in y-direction],
             - theta [range (pi/2, -pi/2), pi/2 points in z-direction]
    """
    x, y, z = np.asarray(v)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, (x * x + y * y) ** .5)
    return phi, theta


def ang2vec(phi, theta):
    """
    Get vector from spherical angles (phi, theta)

    :param phi: rannge (pi, -pi), 0 points in x-direction, pi/2 in y-direction
    :param theta: range (pi/2, -pi/2), pi/2 points in z-direction
    :return: vector of shape (3, n)
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z])


def sph_unit_vectors(phi, theta):
    """
    Get spherical unit vectors e_r, e_theta, e_phi from spherical angles

    :param phi: range (pi, -pi), 0 points in x-direction, pi/2 in y-direction
    :param theta: range (pi/2, -pi/2), pi/2 points in z-direction
    :return: shperical unit vectors e_r, e_theta, e_phi
    """
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)

    return np.array([[ct * cp, st * cp, -sp],
                     [ct * sp, st * sp, cp],
                     [st, -ct, np.zeros_like(phi)]])


def rotation_matrix(rotation_axis, rotation_angle):
    """
    Rotation matrix for given rotation axis and angle.
    See http://en.wikipedia.org/wiki/Euler-Rodrigues_parameters

    :param rotation_axis: rotation axis of shape np.array([x, y, z])
    :param rotation_angle: rotation angle in radians
    :return: rotation matrix R

    Example:
    R = rotation_matrix( np.array([4,4,1]), 1.2 )
    v1 = np.array([3,5,0])
    v2 = np.dot(R, v1)
    """
    rotation_axis = normed(rotation_axis)
    a = np.cos(rotation_angle / 2.)
    b, c, d = - rotation_axis * np.sin(rotation_angle / 2.)
    r = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                  [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                  [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    return np.squeeze(r)


def rotate(v, rotation_axis, rotation_angle):
    """
    Perform rotation for a given rotation axis and angle.

    :param v: vector that is supposed to be rotated in shape np.array([x, y, z])
    :param rotation_axis: rotation axis of shape np.array([x, y, z])
    :param rotation_angle: rotation angle in radians
    :return: rotated vector
    """
    if np.ndim(rotation_axis) > 1:
        raise Exception('rotate: can only take a single rotation axis')
    r = rotation_matrix(rotation_axis, rotation_angle)
    return np.dot(r, v)


def exposure_equatorial(dec, a0=-35.25, zmax=60):
    """
    Relative exposure per solid angle of a detector at latitude a0 (-90, 90 degrees, default: Auger),
    with maximum acceptance zenith angle zmax (0, 90 degrees, default: 60) and for given equatorial declination
    dec (-pi/2, pi/2). See astro-ph/0004016

    :param dec: value(s) of declination in radians (-pi/2, pi/2)
    :param a0: latitude of detector (-90, 90) degrees (default: Auger)
    :param zmax: maximum acceptance zenith angle (0, 90) degrees
    :return: exposure value(s) for input declination value(s) normalized to maximum value of 1
    """
    dec = np.array(dec)
    # noinspection PyTypeChecker,PyUnresolvedReferences
    if (abs(dec) > np.pi / 2).any():
        raise Exception('exposure_equatorial: declination not in range (-pi/2, pi/2)')
    if (zmax < 0) or (zmax > 90):
        raise Exception('exposure_equatorial: zmax not in range (0, 90 degrees)')
    if (a0 < -90) or (a0 > 90):
        raise Exception('exposure_equatorial: a0 not in range (-90, 90 degrees)')

    zmax *= np.pi / 180
    a0 *= np.pi / 180

    xi = (np.cos(zmax) - np.sin(a0) * np.sin(dec)) / np.cos(a0) / np.cos(dec)
    xi = np.clip(xi, -1, 1)
    am = np.arccos(xi)

    cov = np.cos(a0) * np.cos(dec) * np.sin(am) + am * np.sin(a0) * np.sin(dec)
    return cov / np.pi  # normalize the maximum possible value to 1


def rand_phi(n=1):
    """
    Random uniform phi (-pi, pi).

    :param n: number of samples that are drawn
    :return: random numbers in range (-pi, pi)
    """
    return (np.random.rand(n) * 2 - 1) * np.pi


def rand_theta(n=1):
    """
    Random theta (pi/2, -pi/2) from uniform cos(theta) distribution.

    :param n: number of samples that are drawn
    :return: random theta in range (-pi/2, pi/2) from cos(theta) distribution
    """
    return np.pi / 2 - np.arccos(np.random.rand(n) * 2 - 1)


def rand_vec(n=1):
    """
    Random spherical unit vectors.

    :param n: number of vectors that are drawn
    :return: random unit vectors of shape (3, n)
    """
    return ang2vec(rand_phi(n), rand_theta(n))


def rand_fisher(kappa, n=1):
    """
    Random angles to the center of a Fisher distribution with concentration parameter kappa.

    :param kappa: concentration parameter, translates to 1/sigma^2 (sigma: smearing angle in radians)
    :param n: number of vectors drawn from fisher distribution
    :return: theta values (angle towards the mean direction)
    """
    return np.arccos(1 + np.log(1 - np.random.rand(n) * (1 - np.exp(-2 * kappa))) / kappa)


def rand_fisher_vec(vmean, kappa, n=1):
    """
    Random Fisher distributed vectors with mean direction vmean and concentration parameter kappa.

    :param vmean: mean direction of the fisher distribution of shape array([x, y, z])
    :param kappa: concentration parameter, translates to 1/sigma^2 (sigma: smearing angle in radians)
    :param n: number of vectors drawn from fisher distribution
    :return: vectors from fisher distribution of shape (3, n)
    """
    if np.ndim(vmean) > 1:
        raise Exception('rand_fisher_vec: can only take a single mean direction vector')

    # create random directions around (0,0,1)
    t = np.pi / 2 - rand_fisher(kappa, n)
    p = rand_phi(n)
    v = ang2vec(p, t)

    # check for border cases: vmean == +-(0, 0, 1)
    a = angle(vmean, (0, 0, 1))
    if a == 0:
        return v
    if a == np.pi:
        return -v  # pylint: disable=E1130

    # else, rotate (0,0,1) to vmean
    rot_axis = np.cross(vmean, (0, 0, 1))
    rot_angle = angle(vmean, (0, 0, 1))

    return rotate(v, rot_axis, rot_angle)
