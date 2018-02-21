import unittest

import numpy as np
from astrotools import coord, healpytools as hpt

__author__ = 'Marcus Wirtz'


class TestConversions(unittest.TestCase):

    def test_01_pix2ang(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        pix = np.random.randint(0, npix, stat)
        phi, theta = hpt.pix2ang(nside, pix)
        phi_range = (phi >= -np.pi).all() and (phi <= np.pi).all()
        theta_range = (theta >= -np.pi / 2.).all() and (theta <= np.pi / 2.).all()
        self.assertTrue(phi_range and theta_range)

    def test_02_pix2vec(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        pix = np.random.randint(0, npix, stat)
        vec = np.array(hpt.pix2vec(nside, pix))
        self.assertAlmostEqual(np.sum(vec**2, axis=0).all(), np.ones(stat).all())

    def test_03_ang2pix(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        lon = -np.pi + 2 * np.pi * np.random.rand(stat)
        lat_up = np.pi / 6. + 1./3. * np.pi * np.random.rand(stat)
        lat_low = -np.pi / 2. + 1./3. * np.pi * np.random.rand(stat)
        pix_up = hpt.ang2pix(nside, lat_up, lon)
        pix_low = hpt.ang2pix(nside, lat_low, lon)
        up_range = (pix_up >= 0).sum() and (pix_up < int(npix / 2.)).sum()
        low_range = (pix_low < npix).sum() and (pix_low > int(npix / 2.)).sum()
        self.assertTrue(low_range and up_range)

    def test_04_vec2pix(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        vec_up = -1 + 2 * np.random.rand(3 * stat).reshape((3, stat))
        vec_low = -1 + 2 * np.random.rand(3 * stat).reshape((3, stat))
        vec_up[2, :] = 0.1 + np.random.rand(stat)
        vec_low[2, :] = -0.1 - np.random.rand(stat)
        pix_up = hpt.vec2pix(nside, *vec_up)
        pix_low = hpt.vec2pix(nside, *vec_low)
        up_range = (pix_up >= 0).sum() and (pix_up < int(npix / 2.)).sum()
        low_range = (pix_low < npix).sum() and (pix_low > int(npix / 2.)).sum()
        self.assertTrue(low_range and up_range)


class TestPDFs(unittest.TestCase):

    def test_01_exposure(self):
        nside = 64
        exposure = hpt.exposure_pdf(nside)
        self.assertAlmostEqual(np.sum(exposure), 1.)

    def test_02_fisher(self):
        nside = 64
        npix = hpt.nside2npix(nside)
        kappa = 350.
        vmax = np.array([1, 1, 1])
        pix_max = hpt.vec2pix(nside, *vmax)
        pixels, weights = hpt.fisher_pdf(nside, *vmax, k=kappa)
        self.assertEqual(len(pixels), len(weights))
        fisher_map = np.zeros(npix)
        fisher_map[pixels] = weights
        self.assertTrue(np.allclose(np.array([pix_max, 1.]), np.array([np.argmax(fisher_map), np.sum(fisher_map)])))

    def test_03_dipole(self):
        nside = 64
        npix = hpt.nside2npix(nside)
        a = 0.5
        vmax = np.array([1, 1, 1])
        pix_max = hpt.vec2pix(nside, *vmax)
        dipole = hpt.dipole_pdf(nside, a, *vmax)
        self.assertTrue(np.allclose(np.array([pix_max, npix]), np.array([np.argmax(dipole), np.sum(dipole)])))

    def test_04_fisher_delta_small(self):
        """ Fisher distribution has problems for too small angles """
        nside = 64
        deltas = 10.**np.arange(-10, 0, 1)
        vmax = np.array([1, 1, 1])
        for delta in deltas:
            kappa = 1. / delta**2
            pixels, weights = hpt.fisher_pdf(nside, *vmax, k=kappa)
            self.assertTrue(len(pixels) > 0)


class UsefulFunctions(unittest.TestCase):

    def test_01_rand_pix_from_map(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        a = 0.5
        vmax = np.array([1, 1, 1])
        dipole = hpt.dipole_pdf(nside, a, *vmax)
        pixel = hpt.rand_pix_from_map(dipole, stat)
        self.assertTrue((pixel >= 0).sum() and (pixel < npix).sum())

    def test_02_rand_vec_in_pix(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        pix = np.random.randint(0, npix, stat)
        vecs = hpt.rand_vec_in_pix(nside, pix)
        pix_check = hpt.vec2pix(nside, *vecs)
        vecs_check = hpt.pix2vec(nside, pix)
        self.assertTrue((vecs != vecs_check).all() and (pix == pix_check).all())

    def test_03_angle(self):
        stat = 100
        nside = 64
        npix = hpt.nside2npix(nside)
        ipix = np.random.randint(0, npix, stat)
        jpix = np.random.randint(0, npix, stat)
        ivec = hpt.pix2vec(nside, ipix)
        jvec = hpt.pix2vec(nside, jpix)
        angles = hpt.angle(nside, ipix, jpix)
        from astrotools import coord
        angles_coord = coord.angle(ivec, jvec)
        self.assertTrue(np.allclose(angles, angles_coord))

    def test_04_skymap_mean_quantile(self):
        nside = 64
        npix = hpt.nside2npix(nside)
        pix_center = hpt.vec2pix(nside, 1, 0, 0)
        ratio = []
        for ang in np.arange(5, 35, 5):
            delta = np.radians(ang)
            kappa = 1. / delta**2
            pixels, weights = hpt.fisher_pdf(nside, 1, 0, 0, kappa)
            fisher_map = np.zeros(hpt.nside2npix(nside))
            fisher_map[pixels] = weights
            v, alpha = hpt.skymap_mean_quantile(fisher_map)
            ratio.append(alpha / delta)

            self.assertTrue(coord.angle(v, np.array([1, 0, 0]))[0] < 0.01)
            mask = hpt.angle(nside, pix_center, np.arange(npix)) < alpha
            self.assertTrue(np.abs(np.sum(fisher_map[mask]) - 0.68) < 0.1)
        # delta of fisher distribution increases linear with alpha (68 quantil)
        self.assertTrue(np.std(ratio) < 0.05)


if __name__ == '__main__':
    unittest.main()
