import unittest
import os
import numpy as np
from scipy import sparse
from astrotools import gamale, healpytools as hpt

path = os.path.dirname(os.path.realpath(__file__))
lens_path = path + '/toy-lens/jf12-regular.cfg'
nside = 4   # nside resolution of the test toy-lens
npix = hpt.nside2npix(nside)
stat = 10   # statistic (per pixel) of the test toy-lens
lens_bins = np.linspace(17, 20.48, 175)
dlE = (lens_bins[1] - lens_bins[0]) / 2.

test_bins = [100, 150]

class TestLens(unittest.TestCase):

    def test_01_load_and_dimensions(self):
        """ Test raw mldat matrices with simple load function"""
        for bin_t in test_bins:
            mat = gamale.load_lens_part(path + '/toy-lens/jf12-regular-%d.mldat' % bin_t)
            # Sparse matrix that maps npix_extragalactic to npix_observed:
            self.assertTrue(mat.shape == (npix, npix))
            self.assertTrue(int(gamale.max_row_sum(mat)) == stat)
            self.assertTrue(gamale.max_column_sum(mat) >= gamale.max_row_sum(mat))
            # Lower energy bins have higher flux differences (see e.g. arXiv:1607.01645), thus:
            if bin_t > test_bins[0]:
                self.assertTrue(gamale.max_column_sum(mat) < old_mcs)
            old_mcs = gamale.max_column_sum(mat)

    def test_02_lens_class_init(self):
        """ Test lens class with load function"""
        lens = gamale.Lens(lens_path)
        self.assertTrue(lens.nside == nside)
        for i, bin_t in enumerate(test_bins):
            self.assertTrue(os.path.isfile(lens.lens_parts[i]))
            self.assertAlmostEqual(lens.log10r_mins[i], lens_bins[bin_t] - dlE, places=3)
            self.assertAlmostEqual(lens.log10r_max[i], lens_bins[bin_t] + dlE, places=3)
        vec_in = np.random.random(npix)
        self.assertTrue(np.array_equal(lens.neutral_lens_part.dot(vec_in), vec_in))
        self.assertTrue(len(lens.max_column_sum) == len(test_bins))

    def test_03_lens_class_load(self):
        """ Test lens class with load function"""
        lens = gamale.Lens(lens_path)
        for i, bin_t in enumerate(test_bins):
            lp = lens.get_lens_part(lens_bins[bin_t], cache=False)
            self.assertTrue(lens.check_lens_part(lp))
            self.assertTrue(not isinstance(lens.lens_parts[i], sparse.csc.csc_matrix))
            lp = lens.get_lens_part(lens_bins[bin_t], cache=True)
            self.assertTrue(lens.check_lens_part(lp))
            self.assertTrue(isinstance(lens.lens_parts[i], sparse.csc.csc_matrix))

    def test_04_energy_borders(self):
        """ Test energy borders of load function"""
        lens = gamale.Lens(lens_path)
        for i, bini in enumerate(lens_bins):
            try:
                mat = lens.get_lens_part(bini + np.random.uniform(-dlE, dlE))
            except ValueError:
                if i in test_bins:
                    raise Exception("Bin %d was unable to load." % i)
                pass

    def test_05_mean_deflection(self):
        """ Test for higher deflections in lower energy bins """
        for bin_t in test_bins:
            mat = gamale.load_lens_part(path + '/toy-lens/jf12-regular-%d.mldat' % bin_t)
            deflection = gamale.mean_deflection(mat)
            self.assertTrue(deflection >= 0)
            self.assertTrue(deflection < np.pi)
            if bin_t > test_bins[0]:
                self.assertTrue(deflection < old_def)
            old_def = deflection

    def test_06_transform_mean(self):
        """  """
        for bin_t in test_bins:
            lens = gamale.Lens(lens_path)
            lp = lens.get_lens_part(lens_bins[bin_t])
            x, y, z, p, delta = gamale.transform_pix_mean(lp, 0)
            self.assertEqual(x**2 + y**2 + z**2, 1.)


if __name__ == '__main__':
    unittest.main()
