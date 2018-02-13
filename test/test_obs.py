import unittest

import numpy as np
from astrotools import obs, coord
from astrotools import healpytools as hpt


class TestThrust(unittest.TestCase):

    def test_01_point(self):
        p = np.array([1., 0., 0.])[:, np.newaxis]
        T, N = obs.thrust(p, weights=None)
        self.assertTrue(T[0] == 1.)
        self.assertTrue(T[1] == 0.)
        self.assertTrue(T[2] == 0.)

    def test_02_line(self):
        ncrs = 1000
        roi_size = 0.25
        lat = np.zeros(ncrs)
        lon = np.linspace(-roi_size, roi_size, ncrs)
        p = coord.ang2vec(lon, lat)
        T, N = obs.thrust(p, weights=None)
        self.assertTrue(np.abs(T[1] - 0.5 * roi_size) < 1e-3)
        self.assertTrue(np.abs(T[2]) < 1e-3)

    def test_03_iso(self):
        nside = 256
        npix = hpt.nside2npix(nside)
        ncrs = 1000
        roipix = hpt.ang2pix(nside, 0, 0)
        roi_size = 0.25

        angles_pix_to_roi = hpt.angle(nside, roipix, np.arange(0, npix, 1))
        iso_map = np.zeros(npix)
        iso_map[angles_pix_to_roi < roi_size] = 1

        np.random.seed(0)
        p = np.cumsum(iso_map)
        pix = p.searchsorted(np.random.rand(ncrs) * p[-1])

        p = hpt.rand_vec_in_pix(nside, pix)
        T, N = obs.thrust(p, weights=None)
        self.assertTrue(np.abs(T[1] - 4./(3. * np.pi) * roi_size) < 1e-2)
        self.assertTrue(T[2] < T[1])
        self.assertTrue(np.abs(T[2] - T[1]) < 1e-2)


if __name__ == '__main__':
    unittest.main()
