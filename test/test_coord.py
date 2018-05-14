import unittest

import numpy as np
import datetime
from astrotools import coord

__author__ = 'Marcus Wirtz'
stat = 10
np.random.seed(0)


class TestConversions(unittest.TestCase):

    def test_01_eq2gal(self):
        vec_eq = -0.5 + np.random.random((3, stat))
        vec_eq /= np.sqrt(np.sum(vec_eq**2, axis=0))
        vec_gal = coord.eq2gal(vec_eq)
        bool_eq_gal_same = np.allclose(vec_gal, vec_eq)
        bool_normed = np.allclose(np.sum(vec_gal**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_gal_same)

    def test_02_gal2eq(self):
        vec_gal = -0.5 + np.random.random((3, stat))
        vec_gal /= np.sqrt(np.sum(vec_gal**2, axis=0))
        vec_eq = coord.gal2eq(vec_gal)
        bool_eq_gal_same = np.allclose(vec_gal, vec_eq)
        bool_normed = np.allclose(np.sum(vec_eq**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_gal_same)

    def test_03_sgal2gal(self):
        vec_sgal = -0.5 + np.random.random((3, stat))
        vec_sgal /= np.sqrt(np.sum(vec_sgal**2, axis=0))
        vec_gal = coord.sgal2gal(vec_sgal)
        bool_sgal_gal_same = np.allclose(vec_gal, vec_sgal)
        bool_normed = np.allclose(np.sum(vec_gal**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_sgal_gal_same)

    def test_04_gal2sgal(self):
        vec_gal = -0.5 + np.random.random((3, stat))
        vec_gal /= np.sqrt(np.sum(vec_gal**2, axis=0))
        vec_sgal = coord.gal2sgal(vec_gal)
        bool_sgal_gal_same = np.allclose(vec_gal, vec_sgal)
        bool_normed = np.allclose(np.sum(vec_sgal**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_sgal_gal_same)

    def test_05_eq2ecl(self):
        vec_eq = -0.5 + np.random.random((3, stat))
        vec_eq /= np.sqrt(np.sum(vec_eq**2, axis=0))
        vec_ecl = coord.eq2ecl(vec_eq)
        bool_eq_ecl_same = np.allclose(vec_ecl, vec_eq)
        bool_normed = np.allclose(np.sum(vec_ecl**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_ecl_same)

    def test_06_ecl2eq(self):
        vec_ecl = -0.5 + np.random.random((3, stat))
        vec_ecl /= np.sqrt(np.sum(vec_ecl**2, axis=0))
        vec_eq = coord.ecl2eq(vec_ecl)
        bool_eq_ecl_same = np.allclose(vec_ecl, vec_eq)
        bool_normed = np.allclose(np.sum(vec_eq**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_ecl_same)

    def test_07_dms2rad(self):
        deg = 360 * np.random.rand(stat)
        min = 60 * np.random.rand(stat)
        sec = 60 * np.random.rand(stat)
        rad = coord.dms2rad(deg, min, sec)
        self.assertTrue((rad > 0).all() and (rad < 2 * np.pi).all)

    def test_08_hms2rad(self):
        hour = 24 * np.random.rand(stat)
        min = 60 * np.random.rand(stat)
        sec = 60 * np.random.rand(stat)
        rad = coord.hms2rad(hour, min, sec)
        self.assertTrue((rad > 0).all() and (rad < 2 * np.pi).all)

    def test_09_get_hour_angle(self):
        ra = coord.rand_phi(stat)
        self.assertTrue(np.sum(coord.get_hour_angle(ra, ra) == 0) == stat)
        lst = coord.rand_phi(stat)
        ha = coord.get_hour_angle(ra, lst)
        self.assertTrue(np.sum((ha >= 0) & (ha < 2*np.pi)) == stat)

    def test_10_alt_zen(self):
        alt1, alt2 = 0, np.pi / 2
        self.assertTrue(coord.alt2zen(alt1) == np.pi / 2.)
        self.assertTrue(coord.alt2zen(alt2) == 0.)

    def test_11_julian_day(self):
        year, month, day = 2018, 4, 13  # todays date
        today = 2458222                 # computed from online julian date converter
        self.assertTrue(coord.date_to_julian_day(year, month, day) == today)

    def test_12_greenwich_siderial_time(self):
        # Look up reference from http://www.csgnetwork.com/siderealjuliantimecalc.html
        date = datetime.datetime(2004, 1, 1, 0, 0, 0, 0)
        gst = coord.get_greenwich_siderial_time(date)
        hour, min, sec = 6., 39., 58.602988
        self.assertAlmostEqual(gst, hour + min / 60 + sec / 3600, places=3)

    def test_13_local_siderial_time(self):
        # Look up reference from http://www.csgnetwork.com/siderealjuliantimecalc.html
        date = datetime.datetime(2004, 1, 1, 0, 0, 0, 0)
        gst = coord.get_local_sidereal_time(date, np.pi/2.)
        hour, min, sec = 12., 39., 58.602988
        self.assertAlmostEqual(gst, 2 * np.pi * (hour + min / 60 + sec / 3600) / 24, places=3)


class TestVectorCalculations(unittest.TestCase):

    def test_01_normed(self):
        vecs = np.random.rand(3 * stat).reshape((3, stat)) - 0.5
        vecs = coord.normed(vecs)
        self.assertAlmostEqual(vecs.all(), np.ones(stat).all())

    def test_02_normed(self):
        v1 = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1, 1, 1, -1]])
        v2 = np.array([[0, 0, 0, 0],
                       [0, 0, 1, 0],
                       [1, 0, 0, 1]])
        dis = np.array([0, 1, np.sqrt(2), 2])
        distance = coord.distance(v1, v2)
        self.assertAlmostEqual(dis.all(), distance.all())

    def test_03_angle(self):
        v1 = np.array([[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [1, 1, 1, -1]])
        v2 = np.array([[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [1, 0, 1, 1]])
        ang = np.array([0, np.pi/2., np.pi/4., np.pi])
        angle = coord.angle(v1, v2)
        self.assertAlmostEqual(ang.all(), angle.all())

    def test_04_vec2ang(self):
        v = coord.rand_vec(stat)
        phi, theta = coord.vec2ang(v)
        self.assertTrue((phi >= -np.pi).all() and (phi <= np.pi).all() and
                        (theta >= -np.pi).all() and (theta <= np.pi).all())

    def test_05_ang2vec(self):
        phi = coord.rand_phi(stat)
        theta = coord.rand_theta(stat)
        vec = coord.ang2vec(phi, theta)
        self.assertTrue(np.allclose(np.sum(vec**2, axis=0), np.ones(stat)))
        phi2, theta2 = coord.vec2ang(vec)
        self.assertTrue(np.allclose(phi, phi2))
        self.assertTrue(np.allclose(theta, theta2))

    def test_06_rotate(self):
        v1 = coord.rand_vec(stat)
        rot_axis = np.hstack(coord.rand_vec(1))
        angle = 0.25
        v2 = coord.rotate(v1, rot_axis, angle)
        angles = coord.angle(v1, v2)
        self.assertTrue((angles > 0).all() & (angles <= angle).all())
        # rotate back
        v3 = coord.rotate(v2, rot_axis, -angle)
        v4 = coord.rotate(v2, rot_axis, 2*np.pi - angle)
        self.assertTrue(np.allclose(v1, v3))
        self.assertTrue(np.allclose(v3, v4))

        # when rotating around z-axis and vectors have z=0: all angles have to be 0.25
        rot_axis = np.array([0, 0, 1])
        v1 = coord.ang2vec(coord.rand_phi(stat), np.zeros(stat))
        v2 = coord.rotate(v1, rot_axis, angle)
        angles = coord.angle(v1, v2)
        self.assertTrue((angles > angle - 1e-3).all() & (angles < angle + 1e-3).all())

        # when rotating around z-axis all angles correspond to longitude shift
        angles = 2 * np.pi * np.random.random(stat)
        v1 = coord.rand_vec(stat)
        lon1, lat1 = coord.vec2ang(v1)
        v2 = np.array([coord.rotate(vi, rot_axis, ai) for vi, ai in zip(v1.T, angles)]).T
        lon2, lat2 = coord.vec2ang(v2)
        self.assertTrue(np.allclose(lat1, lat2))
        lon_diff = lon1 - lon2
        lon_diff[lon_diff < 0] += 2 * np.pi
        self.assertTrue(np.allclose(lon_diff, angles))

    def test_07_rand_fisher_vec(self):
        vmean = np.array([0, 0, 1])
        sigma = 0.25
        vecs = coord.rand_fisher_vec(vmean, kappa=1./sigma**2, n=stat)
        angles = coord.angle(vecs, vmean)
        self.assertTrue((angles >= 0).all())
        self.assertTrue((np.mean(angles) > 0.5 * sigma) & (np.mean(angles) < 2. * sigma))
        self.assertTrue((angles < 3*sigma).all())


if __name__ == '__main__':
    unittest.main()
