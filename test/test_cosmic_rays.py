import os
import unittest

import numpy as np

from astrotools.cosmic_rays import CosmicRaysBase, CosmicRaysSets

__author__ = 'Martin Urban'
np.random.seed(0)


class TestCosmicRays(unittest.TestCase):
    def test_01_n_cosmic_rays(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        self.assertEqual(crs.ncrs, ncrs)

    def test_01a_n_cosmic_rays_float(self):
        ncrs = 10.
        crs = CosmicRaysBase(ncrs)
        self.assertEqual(crs.ncrs, int(ncrs))
        ncrs = 10.2
        with self.assertRaises(TypeError):
            CosmicRaysBase(ncrs)

    def test_02_set_energy(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["log10e"] = np.arange(1, ncrs + 1, ncrs)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.log10e() > 0))
        self.assertTrue(np.all(crs["log10e"] > 0))

    def test_03_set_new_element(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["karl"] = np.random.uniform(-10, -1, ncrs)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.karl() < 0))
        self.assertTrue(np.all(crs["karl"] < 0))

    def test_03a_set_new_element_via_set(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs.set("karl", np.random.uniform(-10, -1, ncrs))
        self.assertTrue(np.all(crs.karl() < 0))
        self.assertTrue(np.all(crs["karl"] < 0))

    def test_04_numpy_magic(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["karl"] = np.random.uniform(-10, -1, ncrs)
        crs["log10e"] = np.zeros(ncrs)
        self.assertEqual(len(crs["log10e"][crs["karl"] <= 0]), ncrs)

    def test_05_copy_int(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        key = "an_int"
        crs[key] = 10
        crs2 = CosmicRaysBase(crs)
        crs[key] = -2
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs2[key] == 10))

    def test_06_copy_array(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        key = "an_array"
        array = np.random.random(ncrs)
        crs[key] = array
        crs2 = CosmicRaysBase(crs)
        crs[key] = np.random.random(ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(array, crs2[key]))

    def test_07_setting_an_element_as_list(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        length = np.random.randint(2, 6, ncrs)
        random_idx = np.random.randint(0, ncrs)
        crs["likelihoods"] = [np.random.uniform(1, 10, length[i]) for i in range(ncrs)]
        self.assertEqual(len(crs["likelihoods"][random_idx]), length[random_idx])

    def test_08_saving_and_loading(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        length = np.random.randint(2, 6, ncrs)
        key = "karl"
        crs[key] = [np.random.uniform(1, 10, length[i]) for i in range(ncrs)]
        fname = "/tmp/test.npy"
        crs.save(fname)
        crs3 = CosmicRaysBase(fname)
        # noinspection PyTypeChecker
        os.remove(fname)
        self.assertTrue(np.all([np.all(crs3[key][i] == crs[key][i]) for i in range(ncrs)]))

    def test_09_saving_and_loading_pickle(self):
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
        os.remove(fname)
        # noinspection PyTypeChecker
        self.assertTrue(np.all([np.all(crs3[key][i] == crs[key][i]) for i in range(ncrs)]))
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all([np.all(crs3.karl()[i] == crs.karl()[i]) for i in range(ncrs)]))
        self.assertTrue(crs3[key2] == crs[key2])

    def test_10_start_from_dict(self):
        cosmic_rays_dtype = np.dtype([("log10e", float), ("xmax", float), ("time", str), ("other", object)])
        crs = CosmicRaysBase(cosmic_rays_dtype)
        self.assertEqual(crs.ncrs, 0)

    def test_11_add_crs(self):
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

    def test_11a_add_crs_from_cosmic_rays_class(self):
        ncrs1 = 10
        ncrs2 = 40
        crs1 = CosmicRaysBase(ncrs1)
        crs2 = CosmicRaysBase(ncrs2)
        crs1.add_cosmic_rays(crs2)
        self.assertEqual(crs1.ncrs, ncrs1 + ncrs2)

    def test_12_len(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        # noinspection PyTypeChecker
        self.assertEqual(len(crs), crs.ncrs)

    def test_13_add_new_keys(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["log10e"] = np.zeros(ncrs)
        crs["C_best_fit"] = np.ones(ncrs, dtype=float)
        crs["C_best_fit_object"] = np.ones(ncrs, dtype=[("C_best_fit_object", object)])
        crs["rigidities_fit"] = crs["log10e"]
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["C_best_fit"] == 1))
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["rigidities_fit"] == crs["log10e"]))

    def test_13a_change_via_function(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        crs["log10e"] = np.zeros(ncrs)
        crs.log10e(1)
        self.assertTrue(np.all(crs["log10e"] == 1))
        self.assertTrue(np.all(crs.log10e() == 1))

    def test_14_access_by_id(self):
        ncrs = 10
        idx = 8
        crs = CosmicRaysBase(ncrs)
        # crs["C_best_fit"] = np.ones(ncrs, dtype=[("C_best_fit", np.float64)])
        crs["C_best_fit"] = np.ones(ncrs, dtype=float)
        self.assertEqual(crs[idx]["C_best_fit"], 1)

    def test_15_iteration(self):
        ncrs = 10
        crs = CosmicRaysBase(ncrs)
        key = "C_best_fit"
        crs[key] = np.ones(ncrs)
        for i, cr in enumerate(crs):
            cr[key] = i
            self.assertEqual(cr[key], i)

    @unittest.skipIf(os.path.isfile("/.dockerenv"), "Plotting in Docker environment is not possible!")
    def test_16_plotting(self):
        ncrs = 1000
        crs = CosmicRaysBase(ncrs)
        crs['pixel'] = np.random.randint(0, 49152, ncrs)
        crs['log10e'] = 17. + 2.5 * np.random.random(ncrs)

        fname = "/tmp/energy_spectrum.png"
        crs.plot_energy_spectrum(opath=fname)
        crs.plot_eventmap()
        crs.plot_healpy_map()
        self.assertTrue(os.path.isfile(fname))
        os.remove(fname)

    def test_17_initialize_with_array(self):
        energies = np.array(np.random.uniform(18, 20, 100), dtype=[("Energy", float)])
        crs = CosmicRaysBase(energies)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs.get("Energy") >= 18))
        self.assertTrue("Energy" in crs.cosmic_rays.dtype.names)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.Energy() >= 18))

    def test_17a_initialize_with_2d_array(self):
        ncrs = 100
        energies = np.random.uniform(18, 20, ncrs)
        ids = np.arange(0, ncrs, ncrs)
        crs_arr = np.empty((100,), dtype=np.dtype([("energy", "f"), ("cr_id", "i")]))
        crs_arr["energy"] = energies
        crs_arr["cr_id"] = ids
        crs = CosmicRaysBase(crs_arr)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs.get("energy") >= 18))
        self.assertTrue(np.all(crs.get("cr_id") - np.arange(0, ncrs, ncrs) == 0))
        self.assertTrue("energy" in crs.cosmic_rays.dtype.names)
        self.assertTrue("cr_id" in crs.cosmic_rays.dtype.names)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.energy() >= 18))

    def test_18_combine_keys(self):
        ncrs = 100
        crs = CosmicRaysBase(ncrs)
        self.assertTrue(len(crs.keys()) == 0)
        crs['array'] = np.random.randint(0, 49152, 100)
        crs['ndarray'] = np.random.random((5, 2))
        crs['float'] = 5
        self.assertTrue('ndarray' in crs.keys())
        self.assertTrue('array' in crs.keys())
        self.assertTrue('float' in crs.keys())

    def test_19_keys_available(self):
        ncrs = 100
        crs = CosmicRaysBase(ncrs)
        crs['array'] = np.random.randint(0, 49152, 100)
        crs['ndarray'] = np.random.random((5, 2))
        self.assertTrue('array' in crs.keys())
        self.assertTrue('ndarray' in crs.keys())

    def test_20_enumerate(self):
        n_crs = 100
        crs = CosmicRaysBase(cosmic_rays=n_crs)
        imax1, imax2 = 0, 0
        for i, _ in enumerate(crs):
            imax1 = i + 1
        crs["energy"] = np.random.uniform(0, 1, 100)
        for i, _ in enumerate(crs):
            imax2 = i + 1
        self.assertTrue(imax1 == n_crs)
        self.assertTrue(imax2 == n_crs)

    def test_21_assign_with_list(self):
        crs_list = [1, 2, 3, 4]
        with self.assertRaises(NotImplementedError):
            CosmicRaysBase(crs_list)

    def test_22_numpy_integer(self):
        n = np.int16(64)
        crs = CosmicRaysBase(n)
        self.assertTrue(crs.ncrs == n)

    def test_23_access_non_existing_element(self):
        ncrs = 100
        crs = CosmicRaysBase(ncrs)
        crs['array'] = np.random.randint(0, 49152, 100)
        with self.assertRaises(ValueError):
            crs["non_existing"]

    def test_24_set_unallowed_items(self):
        ncrs = 100
        crs = CosmicRaysBase(ncrs)
        with self.assertRaises(KeyError):
            # case where the user does use the value as key and vice versa
            crs[[1, 2, 3]] = "key"

    def test_25_slicing_base(self):
        ncrs = 100
        crs = CosmicRaysBase(ncrs)
        energy = np.random.random(ncrs)
        crs['energy'] = energy
        crs_sub = crs[energy < 0.5]
        self.assertTrue(hasattr(crs_sub, 'keys'))
        self.assertTrue(len(crs_sub) < ncrs)
        self.assertTrue(len(crs_sub['energy']) == len(crs_sub))

    def test_26_set_unfortunate_length_of_string(self):
        _str = 'hallo'
        ncrs = len(_str)
        crs = CosmicRaysBase(ncrs)
        crs['feature1'] = 2 * _str
        crs['feature2'] = _str
        crs['feature1'] = _str
        self.assertTrue(('feature1') in crs.keys())
        self.assertTrue((crs['feature1'] == _str) and (crs['feature2'] == _str))
        self.assertTrue(('feature2') in crs.keys())


class TestCosmicRaysSets(unittest.TestCase):
    def test_01_create(self):
        ncrs = 10
        nsets = 15
        crsset = CosmicRaysSets((nsets, ncrs))
        self.assertEqual(crsset.ncrs, ncrs)
        self.assertEqual(crsset.nsets, nsets)

    def test_01a_create_with_None(self):
        with self.assertRaises(NotImplementedError):
            CosmicRaysSets(None)

    def test_01b_create_with_fake_object(self):
        class test:
            def __init__(self):
                self.type = "CosmicRaysSet"
        with self.assertRaises(AttributeError):
            t = test()
            CosmicRaysSets(t)

    def test_02_get_element_from_set(self):
        ncrs = 10
        nsets = 15
        crsset = CosmicRaysSets((nsets, ncrs))
        # noinspection PyTypeChecker
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crsset["log10e"] == 0.))
        self.assertEqual(crsset["log10e"].shape, (nsets, ncrs))

    def test_03_set_element(self):
        ncrs = 10
        nsets = 15
        crsset = CosmicRaysSets((nsets, ncrs))
        energies = np.random.uniform(18, 20, size=(nsets, ncrs))
        crsset["log10e"] = energies
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crsset["log10e"] >= 18))

    def test_04_get_set_by_number(self):
        ncrs = 10
        nsets = 15
        set_number = 3
        crsset = CosmicRaysSets((nsets, ncrs))
        crsset["creator"] = "Martin"
        subset = crsset[set_number]
        self.assertTrue(len(subset), ncrs)
        self.assertTrue(subset["creator"], "Martin")
        self.assertTrue(len(subset.cosmic_rays), ncrs)

    # noinspection PyTypeChecker
    def test_05_set_in_subset(self):
        ncrs = 10
        nsets = 15
        set_number = 3
        crsset = CosmicRaysSets((nsets, ncrs))
        crsset["creator"] = "Martin"
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        subset = crsset[set_number]
        subset["log10e"] = np.random.uniform(18, 20, ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(subset["log10e"] >= 18))
        idx_begin = int(ncrs * set_number)
        idx_end = int(ncrs * (set_number + 1))
        self.assertTrue(np.all(crsset.cosmic_rays[idx_begin:idx_end]["log10e"] >= 18))
        self.assertTrue(np.all(crsset.cosmic_rays[0:idx_begin]["log10e"] == 0))
        self.assertTrue(np.all(crsset.cosmic_rays[idx_end:]["log10e"] == 0))

    def test_06_copy(self):
        ncrs = 10
        nsets = 15
        crs = CosmicRaysSets((nsets, ncrs))
        key = "an_int"
        crs[key] = 10
        crs2 = CosmicRaysSets(crs)
        crs[key] = -2
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs2[key] == 10))

    def test_07_copy_array(self):
        ncrs = 10
        nsets = 15
        crs = CosmicRaysSets((nsets, ncrs))
        key = "an_array"
        array = np.random.random((nsets, ncrs))
        crs[key] = array
        crs2 = CosmicRaysSets(crs)
        crs[key] = np.random.random((nsets, ncrs))
        # noinspection PyTypeChecker
        self.assertTrue(key not in crs.general_object_store.keys())
        self.assertTrue(key not in crs2.general_object_store.keys())
        self.assertTrue(np.allclose(array, crs2[key]))

    def test_08_copy_gos_array(self):
        # gos = general object store
        ncrs = 10
        nsets = 15
        crs = CosmicRaysSets((nsets, ncrs))
        key = "an_array"
        array = np.random.random(ncrs)
        crs[key] = array
        crs2 = CosmicRaysSets(crs)
        crs[key] = np.random.random(ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(key in crs.general_object_store.keys())
        self.assertTrue(key in crs2.general_object_store.keys())
        self.assertTrue(np.allclose(array, crs2[key]))

    def test_09_save(self):
        ncrs = 10
        nsets = 15
        outpath = "/tmp/cosmicraysset.pkl"
        crsset = CosmicRaysSets((nsets, ncrs))
        crsset["creator"] = "Marcus"
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        crsset.save(outpath)
        self.assertTrue(os.path.exists(outpath))
        os.remove(outpath)

    def test_09_1_save_load(self):
        ncrs = 10
        nsets = 15
        outpath = "/tmp/cosmicraysset_load1.npz"
        crsset = CosmicRaysSets((nsets, ncrs))
        crsset["creator"] = "Marcus"
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        crsset.save(outpath)

        crsset2 = CosmicRaysSets((nsets, ncrs))
        crsset2.load(outpath)
        os.remove(outpath)
        self.assertTrue("creator" in crsset2.keys())
        self.assertTrue("log10e" in crsset2.keys())
        self.assertTrue(np.allclose(crsset["log10e"], crsset2["log10e"]))

    def test_09_2_save_load(self):
        ncrs = 10
        nsets = 15
        outpath = "/tmp/cosmicraysset_load2.npz"
        crsset = CosmicRaysSets((nsets, ncrs))
        crsset["creator"] = "Marcus"
        crsset.save(outpath)

        crsset2 = CosmicRaysSets(outpath)
        os.remove(outpath)
        self.assertTrue("creator" in crsset2.keys())

    def test_10_create_from_filename(self):
        # Create first the set and save it to file
        ncrs = 10
        nsets = 15
        outpath = "/tmp/cosmicraysset.npz"
        crsset = CosmicRaysSets((nsets, ncrs))
        crsset["creator"] = "Martin"
        crsset["log10e"] = np.ones(shape=crsset.shape)
        crsset.save(outpath)
        # reload the set as a new cosmic rays set
        crsset2 = CosmicRaysSets(outpath)
        os.remove(outpath)
        self.assertTrue(crsset2["creator"] == "Martin")
        self.assertTrue(np.shape(crsset2["log10e"]) == (nsets, ncrs))
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crsset2["log10e"] == 1))

    @unittest.skipIf(os.path.isfile("/.dockerenv"), "Plotting in Docker environment is not possible!")
    def test_11_plot(self):
        nsets, ncrs = 10, 100
        crs = CosmicRaysSets((nsets, ncrs))
        crs['pixel'] = np.random.randint(0, 49152, (10, 100))
        crs['log10e'] = 18. + 2.5 * np.random.random((10, 100))

        crs.plot_eventmap()
        crs.plot_energy_spectrum()
        crs.plot_healpy_map()
        self.assertTrue(True)

    @unittest.skipIf(os.path.isfile("/.dockerenv"), "Plotting in Docker environment is not possible!")
    def test_12_plot_from_loaded_cosmic_rays_set(self):
        nsets, ncrs = 10, 100
        crs = CosmicRaysSets((nsets, ncrs))
        crs['pixel'] = np.random.randint(0, 49152, (10, 100))
        crs['log10e'] = 18. + 2.5 * np.random.random((10, 100))
        fname = "/tmp/test_08.npy"
        crs.save(fname)

        crs3 = CosmicRaysSets(fname)
        crs3.plot_eventmap(opath=fname.replace('.npy', '.png'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname.replace('.npy', '.png')))
        os.remove(fname)

    def test_13_combine_keys(self):
        nsets, ncrs = 10, 100
        crs = CosmicRaysSets((nsets, ncrs))
        crs['ndarray'] = np.random.randint(0, 49152, (10, 100))
        crs['array'] = np.random.random(100)
        crs['float'] = 5.
        self.assertTrue('ndarray' in crs.keys())
        self.assertTrue('array' in crs.keys())
        self.assertTrue('float' in crs.keys())

    def test_14_keys_available(self):
        nsets, ncrs = 10, 100
        crs = CosmicRaysSets((nsets, ncrs))
        crs['ndarray'] = np.random.randint(0, 49152, (10, 100))
        crs['array'] = np.random.random(100)
        crs['float'] = 5.
        self.assertTrue('ndarray' in crs.keys())
        self.assertTrue('array' in crs.keys())
        self.assertTrue('float' in crs.keys())

    def test_15_mask_subset(self):
        nsets, ncrs = 100, 10
        ndarray = np.random.randint(0, 49152, (100, 10))
        array = np.random.random(100)
        crs = CosmicRaysSets((nsets, ncrs))
        crs['ndarray'] = ndarray
        crs['array'] = array
        crs['string'] = 'blubb'
        crs['integer'] = 5
        mask = array < 0.2
        crs_subset = crs[mask]
        self.assertTrue(nsets == crs.nsets)
        self.assertTrue('ndarray' in crs_subset.keys())
        self.assertTrue('array' in crs_subset.keys())
        self.assertTrue((crs_subset.nsets > 10) & (crs_subset.nsets < 30))
        self.assertTrue(np.sum(crs_subset['array'] > 0.2) == 0)
        self.assertEqual(crs_subset['string'], crs['string'])
        self.assertEqual(crs_subset['string'], 'blubb')
        self.assertEqual(crs_subset['integer'], crs['integer'])

    def test_16_indexing_subset(self):
        nsets, ncrs = 100, 10
        ndarray = np.random.randint(0, 49152, (100, 10))
        array = np.random.random(100)
        crs = CosmicRaysSets((nsets, ncrs))
        crs['ndarray'] = ndarray
        crs['array'] = array
        crs['string'] = 'blubb'
        crs['integer'] = 5
        indexing = np.random.choice(np.arange(nsets), 20, replace=False)
        crs_subset = crs[indexing]
        self.assertTrue(nsets == crs.nsets)
        self.assertTrue('ndarray' in crs_subset.keys())
        self.assertTrue('array' in crs_subset.keys())
        self.assertTrue(crs_subset.nsets == 20)
        self.assertTrue(np.allclose(crs_subset['array'], array[indexing]))
        self.assertEqual(crs_subset['string'], crs['string'])
        self.assertEqual(crs_subset['string'], 'blubb')
        self.assertEqual(crs_subset['integer'], crs['integer'])

    def test_17_slicing_subset(self):
        nsets, ncrs = 100, 10
        ndarray = np.random.randint(0, 49152, (100, 10))
        array = np.random.random(100)
        crs = CosmicRaysSets((nsets, ncrs))
        crs['ndarray'] = ndarray
        crs['array'] = array
        crs['string'] = 'blubb'
        crs['integer'] = 5
        low, up = 2, 10
        crs_subset = crs[low:up]
        self.assertTrue(nsets == crs.nsets)
        self.assertTrue('ndarray' in crs_subset.keys())
        self.assertTrue('array' in crs_subset.keys())
        self.assertTrue(crs_subset.nsets == int(up - low))
        self.assertTrue(np.allclose(crs_subset['array'], array[low:up]))
        self.assertEqual(crs_subset['string'], crs['string'])
        self.assertEqual(crs_subset['string'], 'blubb')
        self.assertEqual(crs_subset['integer'], crs['integer'])
        with self.assertRaises(ValueError):
            a = np.zeros(10, dtype=[("a", float)])
            crs[a]

    # def test_18_save_large_number_of_sets(self):
    #     # method taken from: https://stackoverflow.com/questions/4319825/python-unittest-opposite-of-assertraises
    #     def test_save():
    #         try:
    #             nsets, ncrs = 100000, 1500
    #             npix = 49152
    #             crs = CosmicRaysSets((nsets, ncrs))
    #
    #             # fill energies, charges and pixel; fails only for charges
    #
    #             crs['log10e'] = auger.rand_energy_from_auger(nsets * ncrs).reshape((nsets, ncrs))
    #             crs['charge'] = np.ones((nsets, ncrs)) * 6
    #             crs['pixel'] = np.random.choice(npix, (nsets, ncrs), p=np.ones(npix) / float(npix)).astype(np.uint16)
    #
    #             crs.save('/tmp/large_set_test_15')
    #             raise ValueError('Everything ok')
    #         except Exception as e:
    #             raise e
    #     with self.assertRaises(ValueError) as cm:
    #         test_save()
    #     the_exception = cm.exception
    #     self.assertEqual(str(the_exception), "Everything ok")

    def test_19_access_functions(self):
        nsets, ncrs, creator = 100, 10, "Peter"
        crs = CosmicRaysSets((nsets, ncrs))
        self.assertTrue(crs.shape, (nsets, ncrs))
        crs["log10e"] = np.random.random((nsets, ncrs))
        self.assertTrue(crs["log10e"].shape, (nsets, ncrs))
        self.assertTrue(crs.log10e().shape, (nsets, ncrs))
        self.assertTrue(crs.shape, (nsets, ncrs))
        crs["creator"] = creator
        self.assertTrue(crs["creator"], creator)
        self.assertTrue(crs.creator(), creator)
        self.assertTrue(crs.shape, (nsets, ncrs))
        with self.assertRaises(TypeError):
            crs.shape()

    def test_20_failing_get(self):
        nsets, ncrs, creator = 100, 10, "Peter"
        crs = CosmicRaysSets((nsets, ncrs))
        crs["creator"] = creator
        with self.assertRaises((ValueError, TypeError)):
            crs[["test"]]

    def test_21a_create_from_crs_list(self):
        ncrs = 10
        crs1 = CosmicRaysBase(ncrs)
        crs1["log10e"] = np.random.rand(ncrs)
        crs1["name"] = "set1"
        crs2 = CosmicRaysBase(ncrs)
        crs2["log10e"] = np.random.rand(ncrs)
        crs2["name"] = "set2"
        crs_s = [crs1, crs2]
        crs_set = CosmicRaysSets(crs_s)
        self.assertTrue(crs_set.shape == (2, 10))
        self.assertTrue(np.all(crs_set[1]["log10e"] == crs2["log10e"]))
        self.assertTrue(len(crs_set["name"]) == 2)
        self.assertTrue(np.all([crs_set["name"][i] == "set%i" % (i+1) for i in range(2)]))

    def test_21a_create_from_crs_non_cosmic_rays_list(self):
        ncrs = 10
        log10e1 = np.random.rand(ncrs)
        log10e2 = np.random.rand(ncrs)
        crs_s = [log10e1, log10e2]
        with self.assertRaises(TypeError):
            CosmicRaysSets(crs_s)

    def test_21b_create_from_crs_list_unequal_nr_crs(self):
        ncrs1, ncrs2 = 10, 11
        crs1 = CosmicRaysBase(ncrs1)
        crs1["log10e"] = np.random.rand(ncrs1)
        crs1["name"] = "set1"
        crs2 = CosmicRaysBase(ncrs2)
        crs2["log10e"] = np.random.rand(ncrs2)
        crs2["name"] = "set2"
        crs_s = [crs1, crs2]
        with self.assertRaises(ValueError):
            CosmicRaysSets(crs_s)

    def test_21c_create_from_crs_list_unequal_attributes(self):
        ncrs = 10
        crs1 = CosmicRaysBase(ncrs)
        crs1["log10e"] = np.random.rand(ncrs)
        crs1["name"] = "set1"
        crs2 = CosmicRaysBase(ncrs)
        crs2["log10e"] = np.random.rand(ncrs)
        crs2["name"] = "set2"
        crs1["not_in_1"] = "test"
        crs_s = [crs1, crs2]
        with self.assertRaises(AttributeError):
            CosmicRaysSets(crs_s)

    def test_21d_create_from_crs_list_fake_crs(self):
        class Fake:
            def __init__(self):
                self.type = "CosmicRaysSet"

            def __len__(self):
                return 10

        ncrs = 10
        crs1 = CosmicRaysBase(ncrs)
        crs1["log10e"] = np.random.rand(ncrs)
        crs2 = Fake()
        crs_s = [crs1, crs2]
        with self.assertRaises(TypeError):
            CosmicRaysSets(crs_s)

    def test_22_access_non_existing_object(self):
        nsets, ncrs = 10, 100
        crs = CosmicRaysSets((nsets, ncrs))
        crs['ndarray'] = np.random.randint(0, 49152, (10, 100))
        with self.assertRaises(ValueError):
            crs["test"]


if __name__ == '__main__':
    unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestCosmicRays)
    # unittest.TextTestRunner(verbosity=2).run(suite)
