import numpy as np

# expose some coordinate related functions
from numpy import dot, cross, rad2deg, deg2rad
from numpy.linalg import norm


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


def eq2gal(v):
    """
    Rotate equatorial to galactical coordinates (same origin)
    """
    return np.dot(_RGE, np.asarray(v))


def gal2eq(v):
    """
    Rotate galactic to equatorial coordinates (same origin)
    """
    return np.dot(_RGE.transpose(), np.asarray(v))


def gal2sgal(v):
    """
    Rotate galactic to supergalactic coordinates (same origin)
    """
    return np.dot(_RSG, np.asarray(v))


def sgal2gal(v):
    """
    Rotate supergalactic to galactic coordinates (same origin)
    """
    return np.dot(_RSG.transpose(), np.asarray(v))


def ecl2eq(v):
    """
    Rotate ecliptic to equatorial coordinates (same origin)
    """
    return np.dot(_REE, np.asarray(v))


def eq2ecl(v):
    """
    Rotate equatorial to ecliptic coordinates (same origin)
    """
    return np.dot(_REE.transpose(), np.asarray(v))


def dms2rad(degree, minutes, seconds):
    """
    Transform declination (degree, minute, second) to radians
    """
    return np.sign(degree) * (np.fabs(degree) + 1. / 60 * minutes + 1. / 3600 * seconds) / 180. * np.pi


def hms2rad(hour, minutes, seconds):
    """
    Transform right ascension (hour, minute, second) to radians
    """
    return (hour / 12. + minutes / 720. + seconds / 43200.) * np.pi


def get_hour_angle(ra, lst):
    """ returns the hour angle (in radians) for a specific right ascension and local sidereal time """
    return (lst - ra) % (2 * np.pi)


def get_azimuth_altitude(declination, latitude, hour_angle):
    """ Used to convert between equatorial and horizon coordinates.
        all angles are in radians
        Auger convention: azimuth (-pi, pi) with 0 pointing eastwards, pi/2 pointing to the north
    """
    alt = np.arcsin(np.sin(declination) * np.sin(latitude) +
                    np.cos(declination) * np.cos(latitude) * np.cos(hour_angle))
    # suedazimuth (S=0, W=pi/2, N=pi, E=-pi/2):
    az_sued = np.arctan2(np.sin(hour_angle) * np.cos(declination),
                         np.cos(hour_angle) * np.cos(declination) * np.sin(latitude) 
                         - np.sin(declination) * np.cos(latitude))
    az_auger = - (az_sued + np.pi)  # azimuth according to auger convention
    mask = az_auger <= -np.pi
    az_auger[mask] = 2 * np.pi + az_auger[mask]
    return alt, az_auger


def alt2zen(elevation):
    return 0.5 * np.pi - elevation


def eq2altaz(ra, dec, latitude, lst):
    """
    Transforms equatorial to local (altitude, azimuth) coordinates
    input arguments are: right ascension, declination, latitude of observer and
    local sidereal time of observer
    """
    return get_azimuth_altitude(dec, latitude, get_hour_angle(ra, lst))


def auger2altaz(zen_auger, az_auger):
    """
    Transformation of local coordinates in auger convention to local coordinates in north azimuth
    """
    az = (0.5 * np.pi - az_auger) % (2 * np.pi)
    alt = 0.5 + np.pi - zen_auger
    return alt, az


def altaz2hourangledec(alt, az, lat):
    """
    Transforms local coordinates (altitude, azimuth) into equatorial coordinates (hour angle and declination)
    input arguments: altitude (-pi/2...pi/2), azimuth in auger system, latitude and local sidereal time of observer
    returns hour angle and declination
    """
#    az = (0.5 * np.pi - az_auger) % (2 * np.pi)
    az = (0.5 * np.pi - az)
    dec = np.arcsin(np.sin(alt) * np.sin(lat) + np.cos(alt) * np.cos(lat) * np.cos(az))
    cosh = (np.sin(alt) - np.sin(lat) * np.sin(dec)) / (np.cos(lat) * np.cos(dec))
    cosh[cosh > 1] = 1  # here: cosh means cos(hour_angle)
    cosh[cosh < -1] = -1
    hour_angle = np.arccos(cosh)

    mask = np.sin(az) > 0.0
    hour_angle[mask] = 2 * np.pi - hour_angle[mask]
    return hour_angle, dec


def altaz2eq(alt, az, lat, lst):
    """
    Transforms local coordinates (altitude, azimuth) into equatorial coordinates
    input arguments: altitude (-pi/2...pi/2), azimuth in auger system, latitude and local sidereal time of observer
    returns right ascension and declination
    """
    hour_angle, dec = altaz2hourangledec(alt, az, lat)

    ra = lst - hour_angle
    return ra, dec


def date_to_julian_day(my_date):
    """Returns the Julian day number of a date.
    from http://code-highlights.blogspot.de/2013/01/julian-date-in-python.html and
    http://code.activestate.com/recipes/117215/"""
    a = (14 - my_date.month) // 12
    y = my_date.year + 4800 - a
    m = my_date.month + 12 * a - 3
    return my_date.day + ((153 * m + 2) // 5) + 365 * y + y // 4 - y // 100 + y // 400 - 32045


def get_greenwich_siderial_time(time):
    """Convert civil time to (mean, wrt the mean equinox) Greenwich sidereal time.
    uncertainty of not taking the apparent time (wrt true equinox) is less then 0.01 deg
    time must be a datetime object
    adapted from http://infohost.nmt.edu/tcc/help/lang/python/examples/sidereal/ims/SiderealTime-gst.html
    """
    import datetime
    # [ nDays  :=  number of days between January 0.0 and utc ]
    dateOrd = time.toordinal()
    jan1Ord = datetime.date(time.year, 1, 1).toordinal()
    nDays = dateOrd - jan1Ord + 1

    janDT = datetime.datetime(time.year, 1, 1)
    janJD = date_to_julian_day(janDT) - 1.5
    s = janJD - 2415020.0
    t = s / 36525.0
    r = (0.00002581 * t + 2400.051262) * t + 6.6460656
    u = r - 24 * (time.year - 1900)
    factor_B = 24.0 - u

    SIDEREAL_A = 0.0657098
    t0 = (nDays * SIDEREAL_A - factor_B)
    decUTC = time.hour + 1. / 60. * time.minute + 1. / 3600. * (time.second + time.microsecond * 1.e-6)
    SIDEREAL_C = 1.002738
    gst = (decUTC * SIDEREAL_C + t0) % 24.0
    return gst


def get_local_sidereal_time(time, longitude):
    gst = get_greenwich_siderial_time(time)
    gst *= np.pi / 12.
    return (gst + longitude) % (2 * np.pi)


def normed(v):
    """
    Return the normalized (lists of) vectors.
    """
    return np.asarray(v) / np.linalg.norm(v, axis=0)


def distance(v1, v2):
    """
    Linear distance between each pair from two (lists of) vectors.
    """
    return np.linalg.norm(np.asarray(v1) - np.asarray(v2), axis=0)


def angle(v1, v2, each2each=False):
    """
    Angular distance in [rad] for each pair from two (lists of) vectors.
    Use each2each=True to calculate every combination.
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

# def minAngle(v1, v2):
#     """
#     Minimum angle of each vector (x1,y1,z1) to any of vectors (x2,y2,z2).
#     """
#     return angle(x1, y1, z1, x2, y2, z2, each2each=True).min(axis=1)


def vec2ang(v):
    """
    Get spherical angles (phi, theta) from a (list of) vector(s).
    phi (pi, -pi), 0 points in x-direction, pi/2 in y-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    x, y, z = np.asarray(v)
    phi = np.arctan2(y, x)
    theta = np.arctan2(z, (x * x + y * y) ** .5)
    return (phi, theta)


def ang2vec(phi, theta):
    """
    Get vector from spherical angles (phi, theta)
    phi (pi, -pi), 0 points in x-direction, pi/2 in y-direction
    theta (pi/2, -pi/2), pi/2 points in z-direction
    """
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)
    return np.array([x, y, z])


def sphUnitVectors(phi, theta):
    """
    Get spherical unit vectors e_r, e_theta, e_phi from spherical angles
    """
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)

    return np.array([
        [ct * cp, st * cp, -sp],
        [ct * sp, st * sp,  cp],
        [     st,     -ct,   0]])


def rotationMatrix(axis, theta):
    """
    Rotation matrix for given rotation axis and angle.
    See http://en.wikipedia.org/wiki/Euler-Rodrigues_parameters

    Example:
    R = rotationMatrix( np.array([4,4,1]), 1.2 )
    v1 = np.array([3,5,0])
    v2 = np.dot(R, v1)
    """
    axis = normed(axis)
    a = np.cos(theta / 2.)
    b, c, d = -axis * np.sin(theta / 2.)
    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def rotate(v, axis, theta):
    R = rotationMatrix(axis, theta)
    return np.dot(R, v)


def exposureEquatorial(dec, a0=-35.25, zmax=60):
    """
    Relative exposure per solid angle of a detector
        at latitude a0 (-90, 90 degrees, default: Auger),
        with maximum acceptance zenith angle zmax (0, 90 degrees, default: 60)
        and for given equatorial declination dec (-pi/2, pi/2).
    See astro-ph/0004016
    """
    dec = np.array(dec)
    if (abs(dec) > np.pi / 2).any():
        raise Exception('exposureEquatorial: declination not in range (-pi/2, pi/2)')
    if (zmax < 0) or (zmax > 90):
        raise Exception('exposureEquatorial: zmax not in range (0, 90 degrees)')
    if (a0 < -90) or (a0 > 90):
        raise Exception('exposureEquatorial: a0 not in range (-90, 90 degrees)')

    zmax *= np.pi / 180
    a0 *= np.pi / 180

    xi = (np.cos(zmax) - np.sin(a0) * np.sin(dec)) / np.cos(a0) / np.cos(dec)
    xi = np.clip(xi, -1, 1)
    am = np.arccos(xi)

    cov = np.cos(a0) * np.cos(dec) * np.sin(am) + am * np.sin(a0) * np.sin(dec)
    return cov / np.pi  # normalize the maximum possible value to 1


def randDeclination(n=1, a0=-35.25, zmax=60):
    """
    Returns n random equatorial declinations (pi/2, -pi/2) of a detector
        at latitude a0 (-90, 90 degrees, default: Auger),
        with maximum acceptance zenith angle zmax (0, 90 degrees, default: 60)
    See coord.exposureEquatorial
    """
    # sample probability distribution using the rejection technique
    nTry = int(3.3 * n) + 50
    dec = np.arcsin(2 * np.random.rand(nTry) - 1)
    maxVal = 0.58  # FIXME: this only works for Auger latitude and zmax
    accept = exposureEquatorial(dec, a0, zmax) > np.random.rand(nTry) * maxVal
    if sum(accept) < n:
        raise Exception("randDeclination: stochastic failure")
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
    return np.pi / 2 - np.arccos(np.random.rand(n) * 2 - 1)


def randVec(n=1):
    """
    Random spherical unit vectors.
    """
    return ang2vec(randPhi(n), randTheta(n))


def randFisher(kappa, n=1):
    """
    Random number from Fisher distribution with concentration parameter kappa.
    """
    return np.arccos(1 + np.log(1 - np.random.rand(n) * (1 - np.exp(-2 * kappa))) / kappa)


def randFisherVec(vmean, kappa, n=1):
    """
    Random Fisher distributed vectors with mean direction vmean and concentration parameter kappa.
    """
    # create random directions around (0,0,1)
    t = np.pi / 2 - randFisher(kappa, n)
    p = randPhi(n)
    v = ang2vec(p, t)

    # rotate (0,0,1) to vmean
    rot_axis = np.cross((0, 0, 1), vmean)
    rot_angle = angle((0, 0, 1), vmean)
    R = rotationMatrix(rot_axis, rot_angle)

    return np.dot(R, v)
