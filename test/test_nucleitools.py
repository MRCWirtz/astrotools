import unittest

from astrotools.nucleitools import Charge2Mass, Mass2Charge
import numpy as np

__author__ = 'Marcus Wirtz'
np.random.seed(0)


class TestNucleiTools(unittest.TestCase):

    def test_01a_charge2mass_double(self):
        test_int = 4
        self.assertTrue(getattr(Charge2Mass(test_int), 'double')() == 8)
        test_arr = np.random.randint(1, 27, 5)
        test_list = list(test_arr)
        self.assertTrue(getattr(Charge2Mass(test_arr), 'double')().dtype == int)
        self.assertTrue(getattr(Charge2Mass(test_list), 'double')().dtype == int)
        self.assertTrue(np.allclose(getattr(Charge2Mass(test_arr), 'double')(),
                                    getattr(Charge2Mass(test_list), 'double')()))

    def test_01b_mass2charge_double(self):
        test_int = 4
        self.assertTrue(getattr(Mass2Charge(test_int), 'double')() == 2)
        test_arr = np.random.randint(1, 57, 5)
        test_list = list(test_arr)
        self.assertTrue(getattr(Mass2Charge(test_arr), 'double')().dtype == int)
        self.assertTrue(getattr(Mass2Charge(test_list), 'double')().dtype == int)
        self.assertTrue(np.allclose(getattr(Mass2Charge(test_arr), 'double')(),
                                    getattr(Mass2Charge(test_list), 'double')()))

    def test_02_charge2mass_empiric(self):
        test_int = 4
        self.assertTrue(getattr(Charge2Mass(test_int), 'empiric')() == 8)
        test_arr = np.random.randint(1, 27, 5)
        test_list = list(test_arr)
        self.assertTrue(getattr(Charge2Mass(test_arr), 'empiric')().dtype == int)
        self.assertTrue(getattr(Charge2Mass(test_list), 'empiric')().dtype == int)
        self.assertTrue(np.allclose(getattr(Charge2Mass(test_arr), 'empiric')(),
                                    getattr(Charge2Mass(test_list), 'empiric')()))

    def test_03a_charge2mass_stable(self):
        test_int = 4
        self.assertTrue(getattr(Charge2Mass(test_int), 'stable')().dtype == int)
        test_arr = np.random.randint(1, 27, 5)
        test_list = list(test_arr)
        a_arr = getattr(Charge2Mass(test_arr), 'stable')()
        a_list = getattr(Charge2Mass(test_list), 'stable')()
        self.assertTrue((a_arr.dtype == int) & (a_list.dtype == int))
        self.assertTrue(np.all((a_arr >= 1) & (a_arr < 60)) & np.all((a_list >= 1) & (a_list < 60)))

    def test_03b_mass2charge_stable(self):
        test_int = 4
        self.assertTrue(getattr(Mass2Charge(test_int), 'stable')().dtype == int)
        test_arr = np.random.randint(1, 57, 5)
        test_arr[test_arr == 5] += 1    # no stable nuclei for A=5 exists
        test_arr[test_arr == 8] += 1    # no stable nuclei for A=8 exists
        test_list = list(test_arr)
        a_arr = getattr(Mass2Charge(test_arr), 'stable')()
        a_list = getattr(Mass2Charge(test_list), 'stable')()
        self.assertTrue((a_arr.dtype == int) & (a_list.dtype == int))
        self.assertTrue(np.all((a_arr >= 1) & (a_arr < 30)) & np.all((a_list >= 1) & (a_list < 30)))
        charge_rec = getattr(Charge2Mass(a_arr), 'stable')()
        self.assertTrue(np.allclose(test_arr, charge_rec, rtol=0.4))

    def test_04_charge2mass_float(self):

        test_float = 5.2
        a = getattr(Charge2Mass(test_float), 'double')()
        self.assertTrue(a == 2 * test_float)
        self.assertTrue(a.dtype == float)

        test_float_array = 10 * np.random.random(1000)
        a = getattr(Charge2Mass(test_float_array), 'double')()
        self.assertTrue((np.mean(a) > 9) & (np.mean(a) < 11))
        self.assertTrue(type(a) == np.ndarray)

    def test_05_there_and_back(self):

        charges = 1 + 25 * np.random.random(10)
        masses = getattr(Charge2Mass(charges), 'empiric')()
        charges_recover = getattr(Mass2Charge(masses), 'empiric')()
        self.assertTrue(np.allclose(charges, charges_recover, rtol=1e-2))

        charges = np.random.randint(1, 27, 10)
        masses = getattr(Charge2Mass(charges), 'empiric')()
        charges_recover = getattr(Mass2Charge(masses), 'empiric')()
        self.assertTrue(np.array_equal(charges, charges_recover))


if __name__ == '__main__':
    unittest.main()
