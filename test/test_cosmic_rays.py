import unittest

import numpy as np
from astrotools.cosmic_rays import CosmicRaysBase

__author__ = 'Martin Urban'


class TestCosmicRays(unittest.TestCase):
    def test_01_n_cosmic_rays(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        self.assertEqual(crs.ncrs, ncrs)

    def test_02_set_energy(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["log10e"] = np.arange(1, ncrs+1, ncrs)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.log10e() > 0))

    def test_03_set_new_element(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["karl"] = np.random.uniform(-10, -1, ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["karl"] < 0))

    def test_04_numpy_magic(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["karl"] = np.random.uniform(-10, -1, ncrs)
        self.assertEqual(len(crs["log10e"][crs["karl"] <= 0]), ncrs)

    def test_05_copy(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        key = "rig"
        crs[key] = 10
        crs2 = CosmicRaysBase(crs)
        crs[key] = -2
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs2[key] == 10))

    def test_06_setting_an_element_as_list(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        length = np.random.randint(2, 6, ncrs)
        random_idx = np.random.randint(0, ncrs)
        crs["likelihoods"] = [np.random.uniform(1, 10, length[i]) for i in range(ncrs)]
        self.assertEqual(len(crs["likelihoods"][random_idx]), length[random_idx])

    def test_07_saving_and_loading(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        length = np.random.randint(2, 6, ncrs)
        key = "karl"
        crs[key] = [np.random.uniform(1, 10, length[i]) for i in range(ncrs)]
        fname = "/tmp/test.npy"
        crs.save(fname)
        crs3 = CosmicRaysBase(fname)
        # noinspection PyTypeChecker
        self.assertTrue(np.all([np.all(crs3[key][i] == crs[key][i]) for i in range(ncrs)]))

    def test_08_saving_and_loading_pickle(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        length = np.random.randint(2, 6, ncrs)
        key = "karl"
        key2 = "production_date"
        crs[key] = [np.random.uniform(1, 10, length[i]) for i in range(ncrs)]
        crs[key2] = "YYYY-MM-DD-HH-MM-SS"
        fname = "/tmp/test.pkl"
        crs.save(fname)
        crs3 = CosmicRaysBase(fname)
        # noinspection PyTypeChecker
        self.assertTrue(np.all([np.all(crs3[key][i] == crs[key][i]) for i in range(ncrs)]))
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all([np.all(crs3.karl()[i] == crs.karl()[i]) for i in range(ncrs)]))
        self.assertTrue(crs3[key2] == crs[key2])

    def test_09_start_from_dict(self):
        cosmic_rays_dtype = np.dtype([("log10e", float), ("xmax", float), ("time", str), ("other", object)])
        crs = CosmicRaysBase(cosmic_rays_dtype)
        self.assertEqual(crs.ncrs, 0)

    def test_10_add_crs(self):
        cosmic_rays_dtype = np.dtype([("log10e", float), ("xmax", float), ("time", "|S8"), ("other", object)])
        crs = CosmicRaysBase(cosmic_rays_dtype)
        ncrs = 10
        new_crs = np.zeros(shape=ncrs, dtype=[("log10e", float), ("xmax", float), ("time", "|S2")])
        new_crs["log10e"] = np.random.exponential(1, ncrs)
        new_crs["xmax"] = np.random.uniform(800, 900, ncrs)
        new_crs["time"] = ["0"] * ncrs
        crs.add_cosmic_rays(new_crs)
        self.assertEqual(crs.ncrs, ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["time"] == b"0"))
        self.assertEqual(crs["time"].dtype, "|S8")
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["xmax"] > 0))

    def test_11_len(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        # noinspection PyTypeChecker
        self.assertEqual(len(crs), crs.ncrs)

    def test_12_add_new_keys(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        # crs["C_best_fit"] = np.ones(ncrs, dtype=[("C_best_fit", np.float64)])
        crs["C_best_fit"] = np.ones(ncrs, dtype=float)
        crs["C_best_fit_object"] = np.ones(ncrs, dtype=[("C_best_fit_object", object)])
        crs["rigidities_fit"] = crs["log10e"]
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["C_best_fit"] == 1))
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["rigidities_fit"] == crs["log10e"]))

    def test_13_access_by_id(self):
        ncrs = 10
        idx = 8
        crs = CosmicRaysBase(ncrs)
        # crs["C_best_fit"] = np.ones(ncrs, dtype=[("C_best_fit", np.float64)])
        crs["C_best_fit"] = np.ones(ncrs, dtype=float)
        self.assertEqual(crs[idx]["C_best_fit"], 1)

    def test_14_iteration(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        key = "C_best_fit"
        crs[key] = np.ones(ncrs)
        for i, cr in enumerate(crs):
            cr[key] = i
            self.assertEqual(cr[key], i)


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestCosmicRays)
    # unittest.TextTestRunner(verbosity=2).run(suite)
