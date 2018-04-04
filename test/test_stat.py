import unittest

import numpy as np
from astrotools import stat

__author__ = 'Marcus Wirtz'


class TestStat(unittest.TestCase):

    def test_01_mid(self):
        a = np.array([0.5, 1.5, 4.5])
        mid_a = stat.mid(a)
        self.assertTrue(np.allclose(mid_a, np.array([1., 3.])))

    def test_02_mean_variance(self):
        a = np.random.normal(1., 0.2, 1000)
        m, v = stat.mean_and_variance(a, np.ones(1000))
        self.assertTrue(np.abs(m - 1.) < 0.1)
        self.assertTrue(np.abs(v - 0.2**2) < 0.01)

        m, v = stat.mean_and_variance(a, a)
        self.assertTrue(m > 1)


if __name__ == '__main__':
    unittest.main()
