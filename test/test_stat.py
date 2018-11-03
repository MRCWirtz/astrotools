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
        for n in np.random.randint(1, 100, 10):
            x = np.random.random(n)
            mid_x = stat.mid(x)
            self.assertTrue(mid_x.size == n-1)
            self.assertTrue((mid_x > min(x)).all())
            self.assertTrue((mid_x < max(x)).all())

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

        a = np.random.random(1001)
        q50 = stat.quantile_1d(a, np.ones(1001), quant=0.5)
        self.assertTrue(np.median(a) == q50)
        # weighting of a
        q50_w = stat.quantile_1d(a, a, quant=0.5)
        self.assertTrue(q50_w > 0.6)
        q20 = stat.quantile_1d(a, np.ones(1001), quant=0.2)
        self.assertTrue(np.abs(q20 - 0.2) < 0.1)
        q80 = stat.quantile_1d(a, np.ones(1001), quant=0.8)
        self.assertTrue(np.abs(q80 - 0.8) < 0.1)
        q80_1d = stat.quantile(a, np.ones(1001), quant=0.8)
        self.assertTrue(q80 == q80_1d)

    def test_03c_quantile_axis(self):

        a = np.random.random((10, 101))
        q50 = stat.quantile(a, np.ones(101), quant=0.5)
        self.assertTrue((np.median(a, axis=-1) == q50).all())

    def test_04_median(self):

        a = np.random.normal(1., 0.2, 100)
        med = stat.median(a, np.ones(a.size))
        self.assertAlmostEqual(med, np.median(a))
        self.assertTrue(stat.median(a, a) > med)

    def test_05_binned_mean_variance(self):

        x = np.clip(np.linspace(1, 100, 10000) + np.random.normal(size=10000), 0.1, None)
        y = 0.5 * x + np.sqrt(x) * np.random.normal(size=10000)
        bins = np.arange(0, 110, 10)
        m, v = stat.binned_mean_and_variance(x, y, bins)
        self.assertTrue(np.sum(m[1:] > np.roll(m, 1)[1:]) >= 8)
        self.assertTrue(np.sum(v[1:] > np.roll(v, 1)[1:]) >= 8)

    def test_06_symm_interval_around(self):

        x = np.random.random(10000)
        xl, xr = stat.sym_interval_around(x, 0.6, 0.5)
        self.assertTrue(xl < xr)
        self.assertTrue(np.isclose(0.6 - xl, xr - 0.6, rtol=5e-2))
        self.assertTrue(np.abs(xl - 0.35) < 0.1)


if __name__ == '__main__':
    unittest.main()
