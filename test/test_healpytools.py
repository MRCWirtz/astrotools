import unittest

import sys
import numpy as np
from astrotools import coord, healpytools as hpt

__author__ = 'Marcus Wirtz'
np.random.seed(0)


class TestConversions(unittest.TestCase):

    def setUp(self):
        self.stat = 100
        self.nside = 64
        self.npix = hpt.nside2npix(self.nside)

    def test_01_pix2ang(self):
        pix = np.random.randint(0, self.npix, self.stat)
        phi, theta = hpt.pix2ang(self.nside, pix)
        phi_range = (phi >= -np.pi).all() and (phi <= np.pi).all()
        theta_range = (theta >= -np.pi / 2.).all() and (theta <= np.pi / 2.).all()
        self.assertTrue(phi_range and theta_range)

    def test_02_pix2vec(self):
        pix = np.random.randint(0, self.npix, self.stat)
        vec = hpt.pix2vec(self.nside, pix)
        self.assertAlmostEqual(np.sum(vec**2, axis=0).all(), np.ones(self.stat).all())
        self.assertTrue(np.array_equal(pix, hpt.vec2pix(self.nside, *vec)))

    def test_03_ang2pix(self):
        lon = -np.pi + 2 * np.pi * np.random.rand(self.stat)
        lat_up = np.pi / 6. + 1./3. * np.pi * np.random.rand(self.stat)
        lat_low = -np.pi / 2. + 1./3. * np.pi * np.random.rand(self.stat)
        pix_up = hpt.ang2pix(self.nside, lat_up, lon)
        pix_low = hpt.ang2pix(self.nside, lat_low, lon)
        up_range = (pix_up >= 0).sum() and (pix_up < int(self.npix / 2.)).sum()
        low_range = (pix_low < self.npix).sum() and (pix_low > int(self.npix / 2.)).sum()
        self.assertTrue(low_range and up_range)

    def test_04_vec2pix(self):
        vec_up = -1 + 2 * np.random.rand(3 * self.stat).reshape((3, self.stat))
        vec_low = -1 + 2 * np.random.rand(3 * self.stat).reshape((3, self.stat))
        vec_up[2, :] = 0.1 + np.random.rand(self.stat)
        vec_low[2, :] = -0.1 - np.random.rand(self.stat)
        pix_up = hpt.vec2pix(self.nside, *vec_up)
        pix_low = hpt.vec2pix(self.nside, *vec_low)
        up_range = (pix_up >= 0).sum() and (pix_up < int(self.npix / 2.)).sum()
        low_range = (pix_low < self.npix).sum() and (pix_low > int(self.npix / 2.)).sum()
        self.assertTrue(low_range and up_range)
        pix_up_2 = hpt.vec2pix(self.nside, vec_up)
        pix_low_2 = hpt.vec2pix(self.nside, vec_low)
        self.assertTrue(np.array_equal(pix_up, pix_up_2))
        self.assertTrue(np.array_equal(pix_low, pix_low_2))


class TestPDFs(unittest.TestCase):

    def setUp(self):
        self.nside = 64
        self.npix = hpt.nside2npix(self.nside)

    def test_01_exposure(self):
        exposure = hpt.exposure_pdf(self.nside)
        self.assertAlmostEqual(np.sum(exposure), 1.)

    def test_02_fisher(self):
        kappa = 350.
        vmax = np.array([1, 1, 1])
        fisher_map = hpt.fisher_pdf(self.nside, vmax, k=kappa)
        self.assertEqual(self.npix, fisher_map.size)
        self.assertEqual(np.sum(fisher_map), 1.)
        pix_max = hpt.vec2pix(self.nside, vmax)
        self.assertEqual(pix_max, np.argmax(fisher_map))
        vecs = hpt.pix2vec(self.nside, np.arange(self.npix))
        vecs_mean = np.sum(vecs * fisher_map[None, :], axis=1)
        self.assertEqual(hpt.vec2pix(self.nside, vecs_mean), pix_max)

        pixels, weights = hpt.fisher_pdf(self.nside, vmax, k=kappa, sparse=True)
        self.assertEqual(len(pixels), len(weights))
        self.assertEqual(pixels[np.argmax(weights)], pix_max)

    def test_03_dipole(self):
        a = 0.5
        vmax = np.array([1, 1, 1])
        pix_max = hpt.vec2pix(self.nside, vmax)
        dipole = hpt.dipole_pdf(self.nside, a, vmax, pdf=False)
        self.assertTrue(np.allclose(np.array([pix_max, self.npix]), np.array([np.argmax(dipole), np.sum(dipole)])))
        dipole_v = hpt.dipole_pdf(self.nside, a, vmax, pdf=False)
        self.assertTrue(np.allclose(dipole, dipole_v, rtol=1e-3))

    def test_04_fisher_delta_small(self):
        """ Fisher distribution has problems for too small angles """
        deltas = 10.**np.arange(-10, 0, 1)
        vmax = np.array([1, 1, 1])
        for delta in deltas:
            kappa = 1. / delta**2
            fisher_map = hpt.fisher_pdf(self.nside, vmax, k=kappa)
            self.assertTrue(np.sum(fisher_map) > 0)


class UsefulFunctions(unittest.TestCase):

    def setUp(self):
        self.stat = 100
        self.nside = 64
        self.npix = hpt.nside2npix(self.nside)

    def test_01_rand_pix_from_map(self):
        a = 0.5
        vmax = np.array([1, 1, 1])
        dipole = hpt.dipole_pdf(self.nside, a, *vmax)
        pixel = hpt.rand_pix_from_map(dipole, self.stat)
        self.assertTrue((pixel >= 0).sum() and (pixel < self.npix).sum())

    def test_02_rand_vec_in_pix(self):
        pix = np.random.randint(0, self.npix, self.stat)
        vecs = hpt.rand_vec_in_pix(self.nside, pix)
        pix_check = hpt.vec2pix(self.nside, *vecs)
        vecs_check = hpt.pix2vec(self.nside, pix)
        self.assertTrue((vecs != vecs_check).all() and (pix == pix_check).all())

    def test_03_angle(self):
        ipix = np.random.randint(0, self.npix, self.stat)
        jpix = np.random.randint(0, self.npix, self.stat)
        ivec = hpt.pix2vec(self.nside, ipix)
        jvec = hpt.pix2vec(self.nside, jpix)
        angles = hpt.angle(self.nside, ipix, jpix)
        from astrotools import coord
        angles_coord = coord.angle(ivec, jvec)
        self.assertTrue(np.allclose(angles, angles_coord))

    def test_04_skymap_mean_quantile(self):
        pix_center = hpt.vec2pix(self.nside, 1, 0, 0)
        ratio = []
        for ang in np.arange(5, 35, 5):
            delta = np.radians(ang)
            kappa = 1. / delta**2
            fisher_map = hpt.fisher_pdf(self.nside, 1, 0, 0, kappa)
            v, alpha = hpt.skymap_mean_quantile(fisher_map)
            ratio.append(alpha / delta)

            self.assertTrue(coord.angle(v, np.array([1, 0, 0]))[0] < 0.01)
            mask = hpt.angle(self.nside, pix_center, np.arange(self.npix)) < alpha
            self.assertTrue(np.abs(np.sum(fisher_map[mask]) - 0.68) < 0.1)
        # delta of fisher distribution increases linear with alpha (68 quantil)
        self.assertTrue(np.std(ratio) < 0.05)

    def test_05_rand_vec_from_map(self):

        dipole_pdf = hpt.dipole_pdf(self.nside, 1, 0, 0, 1)
        vecs = hpt.rand_vec_from_map(dipole_pdf, 10000)
        vec_mean = np.mean(np.array(vecs), axis=-1)
        vec_mean /= np.sqrt(np.sum(vec_mean**2))
        self.assertTrue(vec_mean[0] < 0.05)
        self.assertTrue(vec_mean[1] < 0.05)
        self.assertTrue(vec_mean[2] > 0.99)

    def test_06_statistic(self):

        nside = 8
        dipole_pdf = hpt.dipole_pdf(nside, 0.5, 0, 0, 1)
        vecs = hpt.rand_vec_from_map(dipole_pdf, 100000)

        count = hpt.statistic(nside, *vecs, statistics='count')
        self.assertTrue(np.allclose(dipole_pdf / max(dipole_pdf), count / max(count), atol=0.5))
        frequency = hpt.statistic(nside, *vecs, statistics='frequency')
        self.assertTrue(np.allclose(frequency, count / max(count)))
        with self.assertRaises(ValueError):
            hpt.statistic(nside, *vecs, statistics='mean')
        with self.assertRaises(ValueError):
            hpt.statistic(nside, *vecs, statistics='rms')
        with self.assertRaises(NotImplementedError):
            hpt.statistic(nside, *vecs, statistics='std')

        weights = 1. / dipole_pdf[hpt.vec2pix(nside, *vecs)]
        hpt.statistic(nside, *vecs, statistics='mean', vals=weights)
        hpt.statistic(nside, *vecs, statistics='rms', vals=weights)


class PixelTools(unittest.TestCase):

    def test_01_norder_nside_npix(self):
        norder = 4
        self.assertTrue(hpt.isnpixok(hpt.norder2npix(norder)))
        self.assertTrue(hpt.npix2norder(hpt.norder2npix(norder)) == norder)
        with self.assertRaises(ValueError):
            hpt.npix2norder(4.2)

        self.assertTrue(hpt.isnsideok(hpt.norder2nside(norder)))
        self.assertTrue(hpt.nside2norder(hpt.norder2nside(norder)) == norder)
        with self.assertRaises(ValueError):
            hpt.nside2norder(4.2)


class TestPixelFuncHealpy(unittest.TestCase):
    """ This is the unittest of healpys Pixelfunc """

    def setUp(self):
        # data fixture
        self.theta0 = np.pi / 2. - np.array([1.52911759, 0.78550497, 1.57079633, 0.05103658, 3.09055608])
        self.phi0 = np.array([0., 0.78539816, 1.61988371, 0.78539816, 0.78539816])
        self.lon0 = np.degrees(self.phi0)
        self.lat0 = 90. - np.degrees(self.theta0)

    def test_nside2npix(self):
        self.assertEqual(hpt.nside2npix(512), 3145728)
        self.assertEqual(hpt.nside2npix(1024), 12582912)

    def test_nside2resol(self):
        self.assertAlmostEqual(hpt.nside2resol(512, arcmin=True), 6.87097282363)
        self.assertAlmostEqual(hpt.nside2resol(1024, arcmin=True), 3.43548641181)

    def test_nside2pixarea(self):
        self.assertAlmostEqual(hpt.nside2pixarea(512), 3.9947416351188569e-06)

    def test_ang2pix_ring(self):
        # ensure nside = 1 << 23 is correctly calculated
        # by comparing the original theta phi are restored.
        # NOTE: nside needs to be sufficiently large!
        id = hpt.ang2pix(1048576 * 8, self.phi0, self.theta0, nest=False)
        phi1, theta1 = hpt.pix2ang(1048576 * 8, id, nest=False)
        np.testing.assert_array_almost_equal(theta1, self.theta0, decimal=5)
        np.testing.assert_array_almost_equal(phi1, self.phi0, decimal=5)

    def test_ang2pix_ring_outofrange(self):
        # Healpy_Base2 works up to nside = 2**29.
        # Check that a ValueError is raised for nside = 2**30.
        self.assertRaises(
            ValueError, hpt.ang2pix, 1 << 30, self.theta0, self.phi0, nest=False)

    def test_ang2pix_nest(self):
        # ensure nside = 1 << 23 is correctly calculated
        # by comparing the original theta phi are restored.
        # NOTE: nside needs to be sufficiently large!
        # NOTE: with Healpy_Base this will fail because nside
        #       is limited to 1 << 13 with Healpy_Base.
        id = hpt.ang2pix(1048576 * 8, self.phi0, self.theta0, nest=True)
        phi1, theta1 = hpt.pix2ang(1048576 * 8, id, nest=True)
        np.testing.assert_array_almost_equal(theta1, self.theta0)
        np.testing.assert_array_almost_equal(phi1, self.phi0)

    def test_ang2pix_nest_outofrange_doesntcrash(self):
        # Healpy_Base2 works up to nside = 2**29.
        # Check that a ValueError is raised for nside = 2**30.
        self.assertRaises(
            ValueError, hpt.ang2pix, 1 << 30, self.phi0, self.theta0, nest=False)

    @unittest.skipIf(sys.version_info.major == 2, "Python2 healpy can not deal with lonlat keyword")
    def test_get_interp_val(self):
        m = np.arange(12.)
        val0 = hpt.get_interp_val(m, self.theta0, self.phi0)
        val1 = hpt.get_interp_val(m, self.lon0, self.lat0, lonlat=True)
        np.testing.assert_array_almost_equal(val0, val1)

    @unittest.skipIf(sys.version_info.major == 2, "Python2 healpy can not deal with lonlat keyword")
    def test_get_interp_weights(self):
        p0, w0 = (np.array([0, 1, 4, 5]), np.array([1., 0., 0., 0.]))

        # phi not specified, theta assumed to be pixel
        p1, w1 = hpt.get_interp_weights(1, 0)
        np.testing.assert_array_almost_equal(p0, p1)
        np.testing.assert_array_almost_equal(w0, w1)

        # If phi is not specified, lonlat should do nothing
        p1, w1 = hpt.get_interp_weights(1, 0, lonlat=True)
        np.testing.assert_array_almost_equal(p0, p1)
        np.testing.assert_array_almost_equal(w0, w1)

        p0, w0 = (np.array([1, 2, 3, 0]), np.array([0.25, 0.25, 0.25, 0.25]))

        p1, w1 = hpt.get_interp_weights(1, 0, 0)
        np.testing.assert_array_almost_equal(p0, p1)
        np.testing.assert_array_almost_equal(w0, w1)

        p1, w1 = hpt.get_interp_weights(1, 0, 90, lonlat=True)
        np.testing.assert_array_almost_equal(p0, p1)
        np.testing.assert_array_almost_equal(w0, w1)

    @unittest.skipIf(sys.version_info.major == 2, "Python2 healpy can not deal with lonlat keyword")
    def test_get_all_neighbours(self):
        ipix0 = np.array([8, 4, 0, -1, 1, 6, 9, -1])
        ipix1 = hpt.get_all_neighbours(1, np.pi/2, np.pi/2)
        ipix2 = hpt.get_all_neighbours(1, 90, 0, lonlat=True)
        np.testing.assert_array_almost_equal(ipix0, ipix1)
        np.testing.assert_array_almost_equal(ipix0, ipix2)

    def test_fit_dipole(self):
        nside = 32
        npix = hpt.nside2npix(nside)
        d = [0.3, 0.5, 0.2]
        vec = np.transpose(hpt.pix2vec(nside, np.arange(npix)))
        signal = np.dot(vec, d)
        mono, dipole = hpt.fit_dipole(signal)
        self.assertAlmostEqual(mono, 0.)
        self.assertAlmostEqual(d[0], dipole[0])
        self.assertAlmostEqual(d[1], dipole[1])
        self.assertAlmostEqual(d[2], dipole[2])

    def test_boundaries(self):
        """Test whether the boundary shapes look sane"""
        for lgNside in range(1, 5):
            nside = 1 << lgNside
            for pix in range(hpt.nside2npix(nside)):
                for res in range(1, 50, 7):
                    num = 4*res     # Expected number of points
                    for nest in (True, False):
                        points = hpt.boundaries(nside, pix, res, nest=nest)
                        self.assertTrue(points.shape == (3, num))
                        dist = np.linalg.norm(points[:, :num-1] - points[:, 1:])  # distance between points
                        self.assertTrue((dist != 0).all())
                        dmin = np.min(dist)
                        dmax = np.max(dist)
                        self.assertTrue(dmax/dmin <= 2.0)

    def test_ring(self):
        for lgNside in range(1, 5):
            nside = 1 << lgNside
            numPix = hpt.nside2npix(nside)
            numRings = 4*nside - 1  # Expected number of rings
            for nest in (True, False):
                pix = np.arange(numPix)
                ring = hpt.pix2ring(nside, pix, nest=nest)
                self.assertTrue(pix.shape == ring.shape)
                self.assertTrue(len(set(ring)) == numRings)
                if not nest:
                    first = ring[:numPix-1]
                    second = ring[1:]
                    self.assertTrue(np.logical_or(first == second, first == second-1).all())

    def test_accept_ma_allows_only_keywords(self):
        """ Test whether the accept_ma wrapper accepts calls using only keywords."""
        ma = np.zeros(12 * 16 ** 2)
        try:
            hpt.ud_grade(map_in=ma, nside_out=32)
        except IndexError:
            self.fail("IndexError raised")


if __name__ == '__main__':
    unittest.main()
