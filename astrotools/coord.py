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

    :param year: year (measured after christ)
    :param month: month (january: 1, december: 12)
    :param day: day in the month (1 - 31)
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


def get_local_sidereal_time(time, ra):
    """
    Convert civil time to local sidereal time

    :param time: class instance of datetime.date
    :param ra: right ascension in radians
    :return lst: Local sidereal time (in radians)
    """
    gst = get_greenwich_siderial_time(time)
    gst *= 2 * np.pi / 24.
    return (gst + ra) % (2 * np.pi)


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
    Get spherical unit vectors e_r, e_phi, e_theta from spherical angles phi theta.

    :param phi: range (pi, -pi), 0 points in x-direction, pi/2 in y-direction, float or array_like
    :param theta: range (pi/2, -pi/2), pi/2 points in z-direction, float or array_like
    :return: shperical unit vectors e_r, e_phi, e_theta; each with shape (3, N)
    """
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)

    e_r = np.array([ct * cp, ct * sp, st])
    e_phi = np.array([-sp, cp, np.zeros_like(phi)])
    e_theta = np.array([-st * cp, -st * sp, ct])
    return np.array([e_r, e_phi, e_theta])


def rotation_matrix(rotation_axis, rotation_angle):
    """
    Rotation matrix for given rotation axis and angle.
    See http://en.wikipedia.org/wiki/Euler-Rodrigues_parameters

    :param rotation_axis: rotation axis, either np.array([x, y, z]) or ndarray with shape (3, n)
    :param rotation_angle: rotation angle in radians, either float or array size n
    :return: rotation matrix R, either shape (3, 3) or (3, 3, n)

    Example:
    R = rotation_matrix( np.array([4,4,1]), 1.2 )
    v1 = np.array([3,5,0])
    v2 = np.dot(R, v1)
    """
    assert np.ndim(rotation_axis) == np.ndim(rotation_angle) + 1, "Shape of rotation axis and angle do not not match"
    rotation_axis = normed(rotation_axis)
    a = np.cos(rotation_angle / 2.)
    b, c, d = - rotation_axis * np.sin(rotation_angle / 2.)
    r = np.array([[a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
                  [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
                  [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c]])

    return np.squeeze(r)


def rotate(v, rotation_axis, rotation_angle):
    """
    Perform rotation for given rotation axis and angle(s).

    :param v: vector that is supposed to be rotated, either np.array([x, y, z]) or ndarray with shape (3, n)
    :param rotation_axis: rotation axis, either np.array([x, y, z]) or ndarray with shape (3, n)
    :param rotation_angle: rotation angle in radians, either float or array size n
    :return: rotated vector, same shape as input v
    """
    shape = v.shape
    v, rotation_axis, rotation_angle = np.squeeze(v), np.squeeze(rotation_axis), np.squeeze(rotation_angle)
    r = rotation_matrix(rotation_axis, rotation_angle)
    if np.ndim(rotation_axis) == 1:
        return np.dot(r, v).reshape(shape)

    assert (np.shape(v) == np.shape(rotation_axis))
    return np.sum(r * v[np.newaxis], axis=1).reshape(shape)


def rotate_zaxis_to_x(v, x0):
    """
    Transfers the relative orientation between vectors v and the z-axis towards
    v and the reference vectors x0. Mathematically, the scalar products z_axis*v
    before the rotation and x0*v after rotation are the same (see e.g. unit test).

    :param v: vectors that should be rotated, shape: (3) or (3, n)
    :param x0: reference vectors of same shape like v
    """
    # defines rotation axis by the cross-product with z-axis
    u = np.array([x0[1], -x0[0], np.zeros_like(x0[0])])
    u[2, np.sum(u**2, axis=0) < 1e-10] = 1      # fix z-axis itself
    angles = angle(x0, (0, 0, 1))
    return rotate(v, normed(u), angles)


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
    am = np.arccos(np.clip(xi, -1, 1))

    cov = np.cos(a0) * np.cos(dec) * np.sin(am) + am * np.sin(a0) * np.sin(dec)
    return cov / np.pi  # normalize the maximum possible value to 1


def rand_fisher(kappa, n=1):
    """
    Random angles to the center of a Fisher distribution with concentration parameter kappa.

    :param kappa: concentration parameter, kappa=1/sigma^2 (sigma: smearing angle in radians)
                  either float, or np.array (n is set to kappa.shape[0] then)
    :param n: number of vectors drawn from fisher distribution
    :return: theta values (angle towards the mean direction)
    """
    if np.ndim(kappa) > 0:
        n = kappa.shape[0]
    return np.arccos(1 + np.log(1 - np.random.rand(n) * (1 - np.exp(-2 * kappa))) / kappa)


def rand_phi(n=1):
    """
    Random uniform phi (-pi, pi).

    :param n: number of samples that are drawn
    :return: random numbers in range (-pi, pi)
    """
    return (np.random.rand(n) * 2 - 1) * np.pi


def rand_theta(n=1, theta_min=-np.pi/2, theta_max=np.pi/2):
    """
    Random theta (pi/2, -pi/2) from uniform cos(theta) distribution.

    :param n: number of samples that are drawn
    :return: random theta in range (-pi/2, pi/2) from cos(theta) distribution
    """
    assert theta_max > theta_min
    u = np.sin(theta_min) + (np.sin(theta_max) - np.sin(theta_min)) * np.random.rand(n)
    return np.pi / 2 - np.arccos(u)


def rand_theta_plane(n=1):
    """
    Random theta (pi/2, 0) on a planar surface from uniform cos(theta)^2 distribution.

    :param n: number of samples that are drawn
    :return: random theta on plane in range (pi/2, 0)
    """
    return np.pi / 2 - np.arccos(np.sqrt(np.random.rand(n)))


def rand_vec(n=1):
    """
    Random spherical unit vectors.

    :param n: number of vectors that are drawn
    :return: random unit vectors of shape (3, n)
    """
    return ang2vec(rand_phi(n), rand_theta(n))


def rand_vec_on_surface(x0):
    """
    Given unit normal vectors x0 orthogonal on a surface, samples one isotropic
    direction for each given vector x0 from a cos(theta)*sin(theta) distribution

    :param x0: ortogonal unit vector on the surface, shape: (3, N)
    :return: isotropic directions for the respective normal vectors x0
    """
    n = int(x0.size / 3)
    v = ang2vec(rand_phi(n), rand_theta_plane(n))   # produce random vecs on plane through z-axis
    return rotate_zaxis_to_x(v, x0)                 # rotation to respective surface vector x0


def rand_vec_on_sphere(n, r=1):
    """
    Calculates n random positions x and correpsonding directions on the surface of the
    sphere as they would arise from an isotropic distribution of cosmic rays in the universe.

    :param n: number of drawn samples, int
    :param r: size of the sphere, which scales the position vectors, default: 1
    :return: position vectors, directions
    """
    x = rand_vec(n)
    v = rand_vec_on_surface(x)
    return r*x, v


def rand_exposure_vec(a0=-35.25, zmax=60, n=1, res_theta=5000):
    """
    Random vecs following the exposure of an experiment (equatorial coordinates).
    If you need galactic coordinates, call additionaly: coord.eq2gal()
    # this method bins theta and samples from corresponding probabilities as the
    # corresponding probability function is not integratable and invertable

    :param a0: latitude of detector (-90, 90) in degrees (default: Auger)
    :param zmax: maximum acceptance zenith angle (0, 90) degrees
    :param n: number of samples that are drawn
    :param res_theta: resolution of theta, number of bins in (-pi/2, pi/2)
    :return: random unit vectors from exposure of shape (3, n), equatorial coordinates
    """
    eps = 1. / res_theta
    theta_bin = np.linspace(-np.pi/2.+eps, np.pi/2.-eps, num=res_theta)
    p = np.cos(theta_bin) * exposure_equatorial(theta_bin, a0, zmax)
    thetas = np.random.choice(theta_bin, n, p=p/p.sum()) + np.random.uniform(-eps, eps, n)
    return ang2vec(rand_phi(n), thetas)


def rand_fisher_vec(vmean, kappa, n=1):
    """
    Random Fisher distributed vectors with mean direction vmean and concentration parameter kappa.

    :param vmean: mean direction of the fisher distribution, (x, y, z), either shape (3) or (3, n)
    :param kappa: concentration parameter, translates to 1/sigma^2 (sigma: smearing angle in radians)
    :param n: number of vectors drawn from fisher distribution, becomes m if vmean has shape (3, m)
    :return: vectors from fisher distribution of shape (3, n)
    """
    if np.ndim(vmean) > 1:
        n = vmean.shape[1]
    # create random fisher distributed directions around z-axis (0, 0, 1)
    phi = rand_phi(n)
    theta = np.pi / 2 - rand_fisher(kappa, n)
    v = ang2vec(phi, theta)
    return rotate_zaxis_to_x(v, vmean.reshape((3, -1)))  # rotate these to reference vector vmean
