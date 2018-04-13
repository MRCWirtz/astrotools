import unittest

import numpy as np
import datetime
from astrotools import coord

__author__ = 'Marcus Wirtz'


class TestConversions(unittest.TestCase):

    def test_01_eq2gal(self):
        stat = 10
        vec_eq = -0.5 + np.random.random((3, stat))
        vec_eq /= np.sqrt(np.sum(vec_eq**2, axis=0))
        vec_gal = coord.eq2gal(vec_eq)
        bool_eq_gal_same = np.allclose(vec_gal, vec_eq)
        bool_normed = np.allclose(np.sum(vec_gal**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_gal_same)

    def test_02_gal2eq(self):
        stat = 10
        vec_gal = -0.5 + np.random.random((3, stat))
        vec_gal /= np.sqrt(np.sum(vec_gal**2, axis=0))
        vec_eq = coord.gal2eq(vec_gal)
        bool_eq_gal_same = np.allclose(vec_gal, vec_eq)
        bool_normed = np.allclose(np.sum(vec_eq**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_gal_same)

    def test_03_sgal2gal(self):
        stat = 10
        vec_sgal = -0.5 + np.random.random((3, stat))
        vec_sgal /= np.sqrt(np.sum(vec_sgal**2, axis=0))
        vec_gal = coord.sgal2gal(vec_sgal)
        bool_sgal_gal_same = np.allclose(vec_gal, vec_sgal)
        bool_normed = np.allclose(np.sum(vec_gal**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_sgal_gal_same)

    def test_04_gal2sgal(self):
        stat = 10
        vec_gal = -0.5 + np.random.random((3, stat))
        vec_gal /= np.sqrt(np.sum(vec_gal**2, axis=0))
        vec_sgal = coord.gal2sgal(vec_gal)
        bool_sgal_gal_same = np.allclose(vec_gal, vec_sgal)
        bool_normed = np.allclose(np.sum(vec_sgal**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_sgal_gal_same)

    def test_05_eq2ecl(self):
        stat = 10
        vec_eq = -0.5 + np.random.random((3, stat))
        vec_eq /= np.sqrt(np.sum(vec_eq**2, axis=0))
        vec_ecl = coord.eq2ecl(vec_eq)
        bool_eq_ecl_same = np.allclose(vec_ecl, vec_eq)
        bool_normed = np.allclose(np.sum(vec_ecl**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_ecl_same)

    def test_06_ecl2eq(self):
        stat = 10
        vec_ecl = -0.5 + np.random.random((3, stat))
        vec_ecl /= np.sqrt(np.sum(vec_ecl**2, axis=0))
        vec_eq = coord.ecl2eq(vec_ecl)
        bool_eq_ecl_same = np.allclose(vec_ecl, vec_eq)
        bool_normed = np.allclose(np.sum(vec_eq**2, axis=0), np.ones(stat))
        self.assertTrue(bool_normed and not bool_eq_ecl_same)

    def test_07_dms2rad(self):
        stat = 10
        deg = 360 * np.random.rand(stat)
        min = 60 * np.random.rand(stat)
        sec = 60 * np.random.rand(stat)
        rad = coord.dms2rad(deg, min, sec)
        self.assertTrue((rad > 0).all() and (rad < 2 * np.pi).all)

    def test_08_hms2rad(self):
        stat = 10
        hour = 24 * np.random.rand(stat)
        min = 60 * np.random.rand(stat)
        sec = 60 * np.random.rand(stat)
        rad = coord.hms2rad(hour, min, sec)
        self.assertTrue((rad > 0).all() and (rad < 2 * np.pi).all)

    def test_09_get_hour_angle(self):
        stat = 10
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
        stat = 10
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
        stat = 10
        v = coord.rand_vec(stat)
        phi, theta = coord.vec2ang(v)
        self.assertTrue((phi >= -np.pi).all() and (phi <= np.pi).all() and
                        (theta >= -np.pi).all() and (theta <= np.pi).all())

    def test_05_ang2vec(self):
        stat = 10
        phi = coord.rand_phi(stat)
        theta = coord.rand_theta(stat)
        vec = coord.ang2vec(phi, theta)
        self.assertTrue(np.allclose(np.sum(vec**2, axis=0), np.ones(stat)))
        phi2, theta2 = coord.vec2ang(vec)
        self.assertTrue(np.allclose(phi, phi2))
        self.assertTrue(np.allclose(theta, theta2))


if __name__ == '__main__':
    unittest.main()
