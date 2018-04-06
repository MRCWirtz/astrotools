import unittest

import numpy as np
from astrotools import stat

__author__ = 'Marcus Wirtz'
np.random.seed(1)


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

    def test_03a_quantile_gauss(self):

        a = np.random.normal(1., 0.2, 1000)
        q50 = stat.quantile_1d(a, np.ones(1000), quant=0.5)
        self.assertTrue(np.abs(q50 - 1.) < 0.1)
        q10 = stat.quantile_1d(a, np.ones(1000), quant=0.1)
        self.assertTrue(q10 < 0.8)
        q90 = stat.quantile_1d(a, np.ones(1000), quant=0.9)
        self.assertTrue(q90 > 1.2)

    def test_03b_quantile_uniform(self):

        a = np.random.random(1000)
        q50 = stat.quantile_1d(a, np.ones(1000), quant=0.5)
        self.assertTrue(np.abs(q50 - 0.5) < 0.1)
        q50_w = stat.quantile_1d(a, a, quant=0.5)
        self.assertTrue(q50_w > 0.6)
        q20 = stat.quantile_1d(a, np.ones(1000), quant=0.2)
        self.assertTrue(np.abs(q20 - 0.2) < 0.1)
        q80 = stat.quantile_1d(a, np.ones(1000), quant=0.8)
        self.assertTrue(np.abs(q80 - 0.8) < 0.1)

    def test_04_median(self):

        a = np.random.normal(1., 0.2, 100)
        med = stat.median(a, np.ones(a.size))
        self.assertTrue(med == np.median(a))
        self.assertTrue(stat.median(a, a) > med)

    def test_05_binned_mean_variance(self):

        x = np.clip(np.linspace(1, 100, 10000) + np.random.normal(size=10000), 0.1, None)
        y = 0.5 * x + np.sqrt(x) * np.random.normal(size=10000)
        bins = np.arange(0, 110, 10)
        m = stat.binned_mean(x, y, bins)
        self.assertTrue(np.sum(m[1:] > np.roll(m, 1)[1:]) >= 8)

        m2, v2 = stat.binned_mean_and_variance(x, y, bins)
        self.assertTrue(np.allclose(m, m2))
        self.assertTrue(np.sum(v2[1:] > np.roll(v2, 1)[1:]) >= 8)

    def test_06_symm_interval_around(self):

        x = np.random.random(10000)
        xl, xr = stat.sym_interval_around(x, 0.6, 0.5)
        self.assertTrue(xl < xr)
        self.assertTrue(np.isclose(0.6 - xl, xr - 0.6, rtol=5e-2))
        self.assertTrue(np.abs(xl - 0.35) < 0.1)


if __name__ == '__main__':
    unittest.main()
