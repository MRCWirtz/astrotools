from astrotools import coord
import numpy as np


x_sg = np.array([1, 2, 2, 3])
y_sg = np.array([7, 2, 3, 6])
z_sg = np.array([9, 0, -1, 0])

# x_g, y_g, z_g = coord.sgal2gal(x_sg, y_sg, z_sg)



lat = np.deg2rad(-35.1104447)
lon = np.deg2rad(-69.53775804)


for lst in np.linspace(0, np.pi * 2, 20):
    alt = np.linspace(-np.pi * 0.5 + 0.1, np.pi * 0.5 - 0.1, 20)
    az = np.linspace(0, 2 * np.pi, 20)
    M = np.meshgrid(alt, az)
    alts = M[0].flatten()
    azs = M[1].flatten()
    for alt, az in zip(alts, azs):
        ra, dec = coord.altaz2eq2(np.array([alt]), np.array([az]), lat, lst)
        alt2, az2 = coord.eq2altaz(ra, dec, lat, lst)
        if(np.abs((alt - alt2)) > 0.001):
            print
            print "ra dec", ra, dec
            print "alt", alt, alt2, np.abs((alt - alt2))
            print "az", az, az2, np.abs((az - az2))
            print "ra dec", coord.altaz2eq2(alt2, az2, lat, lst)
#         if(np.abs((az - az2) % (2 * np.pi)) > 0.001):
#             print az, az2

# lst = 0
# alt = np.array([0, 0.5, 0.5])
# az = np.array([0.1, 0.1, -0.1 + 2 * np.pi])
# print "alt, az", alt, az
# ra, dec = coord.altaz2eq(alt, az, lat, lst)
# print "ra, dec", ra, dec
# alt, az = coord.eq2altaz(ra, dec, lat, lst)
# print "alt, az", alt, az
# ra, dec = coord.altaz2eq(alt, az, lat, lst)
# print "ra, dec", ra, dec
#
# print
# ra = np.ones(2) * 0.82728975
# decc = 0.95093335
# dec = np.array([decc, np.pi * 2 - decc])
# # dec = np.linspace(-np.pi, np.pi, 10)
# print ra, dec
# alt, az = coord.eq2altaz(ra, dec, lat, lst)
# print alt, az
# ra, dec = coord.altaz2eq(alt, az, lat, lst)
# print ra, dec
