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
        old_mcs = None
        for bin_t in test_bins:
            toy_lens_path = path + '/toy-lens/jf12-regular-%d.mldat' % bin_t
            lp = gamale.load_lens_part(toy_lens_path)
            # Sparse matrix that maps npix_extragalactic to npix_observed:
            self.assertTrue(lp.shape == (npix, npix))
            mrs = lp.sum(axis=1).max()
            mcs = lp.sum(axis=0).max()
            self.assertTrue(int(mrs) == stat)
            self.assertTrue(mcs >= mrs)
            # Lower energy bins have higher flux differences
            # (see e.g. arXiv:1607.01645), thus:
            if bin_t > test_bins[0]:
                self.assertTrue(mcs < old_mcs)
            old_mcs = mcs

    def test_02_lens_class_init(self):
        """ Test lens class with load function"""
        lens = gamale.Lens(lens_path)
        self.assertTrue(lens.nside == nside)
        for i, bin_t in enumerate(test_bins):
            self.assertTrue(os.path.isfile(lens.lens_parts[i]))
            lb = lens_bins[bin_t]
            self.assertAlmostEqual(lens.log10r_mins[i], lb - dlE, places=3)
            self.assertAlmostEqual(lens.log10r_max[i], lb + dlE, places=3)
        vec_in = np.random.random(npix)
        nlp = lens.neutral_lens_part.dot(vec_in)
        self.assertTrue(np.array_equal(nlp, vec_in))
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
                lp = lens.get_lens_part(bini + np.random.uniform(-dlE, dlE))
                lens.check_lens_part(lp)
            except ValueError:
                if i in test_bins:
                    raise Exception("Bin %d was unable to load." % i)
                pass

    def test_05_mean_deflection(self):
        """ Test for higher deflections in lower energy bins """
        old_def = None
        for bin_t in test_bins:
            toy_lens_path = path + '/toy-lens/jf12-regular-%d.mldat' % bin_t
            lp = gamale.load_lens_part(toy_lens_path)
            deflection = gamale.mean_deflection(lp)
            self.assertTrue(deflection >= 0)
            self.assertTrue(deflection < np.pi)
            if bin_t > test_bins[0]:
                self.assertTrue(deflection < old_def)
            old_def = deflection


if __name__ == '__main__':
    unittest.main()
