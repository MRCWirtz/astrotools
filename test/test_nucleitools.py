import unittest

from astrotools.nucleitools import *
import numpy as np

__author__ = 'Marcus Wirtz'


class TestNucleiTools(unittest.TestCase):

    def test_01_charge2mass_double(self):
        test_int = 4
        self.assertTrue(getattr(Charge2Mass(test_int), 'double')() == 8)
        test_arr = np.random.randint(1, 27, 5)
        test_list = list(test_arr)
        self.assertTrue(getattr(Charge2Mass(test_arr), 'double')().dtype == int)
        self.assertTrue(getattr(Charge2Mass(test_list), 'double')().dtype == int)
        self.assertTrue(np.allclose(getattr(Charge2Mass(test_arr), 'double')(),
                                    getattr(Charge2Mass(test_list), 'double')()))

    def test_02_charge2mass_empiric(self):
        test_int = 4
        self.assertTrue(getattr(Charge2Mass(test_int), 'empiric')() == 8)
        test_arr = np.random.randint(1, 27, 5)
        test_list = list(test_arr)
        self.assertTrue(getattr(Charge2Mass(test_arr), 'empiric')().dtype == int)
        self.assertTrue(getattr(Charge2Mass(test_list), 'empiric')().dtype == int)
        self.assertTrue(np.allclose(getattr(Charge2Mass(test_arr), 'empiric')(),
                                    getattr(Charge2Mass(test_list), 'empiric')()))

    def test_03_charge2mass_stable(self):
        test_int = 4
        self.assertTrue(getattr(Charge2Mass(test_int), 'stable')().dtype == int)
        test_arr = np.random.randint(1, 27, 5)
        test_list = list(test_arr)
        a_arr = getattr(Charge2Mass(test_arr), 'stable')()
        a_list = getattr(Charge2Mass(test_list), 'stable')()
        self.assertTrue((a_arr.dtype == int) & (a_list.dtype == int))
        self.assertTrue(np.all((a_arr >= 1) & (a_arr < 60)) & np.all((a_list >= 1) & (a_list < 60)))


if __name__ == '__main__':
    unittest.main()
