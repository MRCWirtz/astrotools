import unittest

import numpy as np
import healpy as hp
from astrotools import healpytools as hpt

__author__ = 'Marcus Wirtz'

class TestConversions(unittest.TestCase):

    def test_01_pix2ang(self):
        stat = 100
        nside = 64
        npix = hp.nside2npix(nside)
        pix = np.random.randint(0, npix, stat)
        phi, theta = hpt.pix2ang(nside, pix)
        phi_range = (phi >= -np.pi).all() and (phi <= np.pi).all()
        theta_range = (theta >= -np.pi / 2.).all() and (theta <= np.pi / 2.).all()
        self.assertTrue(phi_range and theta_range)

    def test_02_pix2vec(self):
        stat = 100
        nside = 64
        npix = hp.nside2npix(nside)
        pix = np.random.randint(0, npix, stat)
        vec = np.array(hpt.pix2vec(nside, pix))
        self.assertAlmostEqual(np.sum(vec**2, axis=0).all(), np.ones(stat).all())

    def test_03_ang2pix(self):
        stat = 100
        nside = 64
        npix = hp.nside2npix(nside)
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
        npix = hp.nside2npix(nside)
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
        npix = hp.nside2npix(nside)
        kappa = 350.
        vmax = np.array([1, 1, 1])
        pix_max = hp.vec2pix(nside, *vmax)
        pixels, weights = hpt.fisher_pdf(nside, *vmax, k=kappa)
        fisher_map = np.zeros(npix)
        fisher_map[pixels] = weights
        self.assertTrue(np.allclose(np.array([pix_max, 1.]), np.array([np.argmax(fisher_map), np.sum(fisher_map)])))

    def test_03_dipole(self):
        nside = 64
        npix = hp.nside2npix(nside)
        a = 0.5
        vmax = np.array([1, 1, 1])
        pix_max = hp.vec2pix(nside, *vmax)
        dipole = hpt.dipole_pdf(nside, a, *vmax)
        self.assertTrue(np.allclose(np.array([pix_max, npix]), np.array([np.argmax(dipole), np.sum(dipole)])))


class UsefulFunctions(unittest.TestCase):

    def test_01_rand_pix_from_map(self):
        stat = 100
        nside = 64
        npix = hp.nside2npix(nside)
        a = 0.5
        vmax = np.array([1, 1, 1])
        dipole = hpt.dipole_pdf(nside, a, *vmax)
        pixel = hpt.rand_pix_from_map(dipole, stat)
        self.assertTrue((pixel >= 0).sum() and (pixel < npix).sum())

    def test_02_rand_vec_in_pix(self):
        stat = 100
        nside = 64
        npix = hp.nside2npix(nside)
        pix = np.random.randint(0, npix, stat)
        vecs = hpt.rand_vec_in_pix(nside, pix)
        pix_check = hp.vec2pix(nside, *vecs)
        vecs_check = hp.pix2vec(nside, pix)
        self.assertTrue((vecs != vecs_check).all() and (pix == pix_check).all())

    def test_03_angle(self):
        stat = 100
        nside = 64
        npix = hp.nside2npix(nside)
        ipix = np.random.randint(0, npix, stat)
        jpix = np.random.randint(0, npix, stat)
        ivec = hp.pix2vec(nside, ipix)
        jvec = hp.pix2vec(nside, jpix)
        angles = hpt.angle(nside, ipix, jpix)
        from astrotools import coord
        angles_coord = coord.angle(ivec, jvec)
        self.assertTrue(np.allclose(angles, angles_coord))

if __name__ == '__main__':
    unittest.main()