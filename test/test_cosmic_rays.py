import os
import unittest

import numpy as np

from astrotools.cosmic_rays import CosmicRaysBase, CosmicRaysSets

__author__ = 'Martin Urban'
user = os.getenv('USER')
np.random.seed(0)


class TestCosmicRays(unittest.TestCase):

    def setUp(self):
        self.ncrs = 10

    def test_01_n_cosmic_rays(self):
        crs = CosmicRaysBase(self.ncrs)
        self.assertEqual(crs.ncrs, self.ncrs)

        crs = CosmicRaysBase(self.ncrs)
        self.assertEqual(crs.ncrs, int(self.ncrs))
        ncrs = 10.2
        with self.assertRaises(TypeError):
            CosmicRaysBase(ncrs)

    def test_02_set_energy(self):
        crs = CosmicRaysBase(self.ncrs)
        crs["log10e"] = np.arange(1, self.ncrs + 1, self.ncrs)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(len(crs.log10e() == self.ncrs))
        self.assertTrue(len(crs["log10e"] == self.ncrs))
        self.assertTrue(np.all(crs.log10e() > 0))
        self.assertTrue(np.all(crs["log10e"] > 0))

    def test_03_set_new_element(self):
        crs = CosmicRaysBase(self.ncrs)
        crs["karl"] = np.random.uniform(-10, -1, self.ncrs)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.karl() < 0))
        self.assertTrue(np.all(crs["karl"] < 0))

        crs = CosmicRaysBase(self.ncrs)
        crs.set("karl", np.random.uniform(-10, -1, self.ncrs))
        self.assertTrue(np.all(crs.karl() < 0))
        self.assertTrue(np.all(crs["karl"] < 0))

    def test_04_numpy_magic(self):
        crs = CosmicRaysBase(self.ncrs)
        crs["karl"] = np.random.uniform(-10, -1, self.ncrs)
        crs["klaus"] = np.linspace(-10, 10, self.ncrs)
        crs["log10e"] = np.zeros(self.ncrs)
        self.assertEqual(len(crs["log10e"][crs["karl"] <= 0]), self.ncrs)
        self.assertEqual(len(crs["log10e"][crs["klaus"] <= 0]), int(self.ncrs / 2))

    def test_05_copy_int(self):
        crs = CosmicRaysBase(self.ncrs)
        key = "an_int"
        crs[key] = 10
        crs2 = CosmicRaysBase(crs)
        crs[key] = -2
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs2[key] == 10))
        self.assertTrue(np.all(crs[key] == -2))

    def test_06_copy_array(self):
        crs = CosmicRaysBase(self.ncrs)
        key = "an_array"
        array = np.random.random(self.ncrs)
        crs[key] = array
        crs2 = CosmicRaysBase(crs)
        crs[key] = np.random.random(self.ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(np.allclose(array, crs2[key]))

    def test_07_setting_an_element_as_list(self):
        crs = CosmicRaysBase(self.ncrs)
        length = np.random.randint(2, 6, self.ncrs)
        random_idx = np.random.randint(0, self.ncrs)
        crs["likelihoods"] = [np.random.uniform(1, 10, length[i]) for i in range(self.ncrs)]
        self.assertEqual(len(crs["likelihoods"][random_idx]), length[random_idx])

    def test_08_saving_and_loading(self):
        crs = CosmicRaysBase(self.ncrs)
        length = np.random.randint(2, 6, self.ncrs)
        key = "karl"
        crs[key] = [np.random.uniform(1, 10, length[i]) for i in range(self.ncrs)]
        fname = "/tmp/test-%s.npy" % user
        crs.save(fname)
        crs3 = CosmicRaysBase(fname)
        # noinspection PyTypeChecker
        os.remove(fname)
        self.assertTrue(np.all([np.all(crs3[key][i] == crs[key][i]) for i in range(self.ncrs)]))

    def test_09_saving_and_loading_pickle(self):
        crs = CosmicRaysBase(self.ncrs)
        length = np.random.randint(2, 6, self.ncrs)
        key = "karl"
        key2 = "production_date"
        crs[key] = [np.random.uniform(1, 10, length[i]) for i in range(self.ncrs)]
        crs[key2] = "YYYY-MM-DD-HH-MM-SS"
        fname = "/tmp/test-%s.pkl" % user
        crs.save(fname)
        crs3 = CosmicRaysBase(fname)
        os.remove(fname)
        # noinspection PyTypeChecker
        self.assertTrue(np.all([np.all(crs3[key][i] == crs[key][i]) for i in range(self.ncrs)]))
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all([np.all(crs3.karl()[i] == crs.karl()[i]) for i in range(self.ncrs)]))
        self.assertTrue(crs3[key2] == crs[key2])

    def test_10_start_from_dict(self):
        cosmic_rays_dtype = np.dtype([("log10e", float), ("xmax", float), ("time", str), ("other", object)])
        crs = CosmicRaysBase(cosmic_rays_dtype)
        self.assertEqual(crs.ncrs, 0)

    def test_11_add_crs(self):
        cosmic_rays_dtype = np.dtype([("log10e", float), ("xmax", float), ("time", "|S8"), ("other", object)])
        crs = CosmicRaysBase(cosmic_rays_dtype)
        new_crs = np.zeros(shape=self.ncrs, dtype=[("log10e", float), ("xmax", float), ("time", "|S2")])
        new_crs["log10e"] = np.random.exponential(1, self.ncrs)
        new_crs["xmax"] = np.random.uniform(800, 900, self.ncrs)
        new_crs["time"] = ["0"] * self.ncrs
        crs.add_cosmic_rays(new_crs)
        self.assertEqual(crs.ncrs, self.ncrs)
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
        crs = CosmicRaysBase(self.ncrs)
        # noinspection PyTypeChecker
        self.assertEqual(len(crs), crs.ncrs)

    def test_13_add_new_keys(self):
        crs = CosmicRaysBase(self.ncrs)
        crs["log10e"] = np.zeros(self.ncrs)
        crs["C_best_fit"] = np.ones(self.ncrs, dtype=float)
        crs["C_best_fit_object"] = np.ones(self.ncrs, dtype=[("C_best_fit_object", object)])
        crs["rigidities_fit"] = crs["log10e"]
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["C_best_fit"] == 1))
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs["rigidities_fit"] == crs["log10e"]))

    def test_13a_change_via_function(self):
        crs = CosmicRaysBase(self.ncrs)
        crs["log10e"] = np.zeros(self.ncrs)
        crs.log10e(1)
        self.assertTrue(np.all(crs["log10e"] == 1))
        self.assertTrue(np.all(crs.log10e() == 1))

    def test_14_access_by_id(self):
        idx = 8
        crs = CosmicRaysBase(self.ncrs)
        # crs["C_best_fit"] = np.ones(ncrs, dtype=[("C_best_fit", np.float64)])
        crs["C_best_fit"] = np.ones(self.ncrs, dtype=float)
        self.assertEqual(crs[idx]["C_best_fit"], 1)

    def test_15_iteration(self):
        crs = CosmicRaysBase(self.ncrs)
        key = "C_best_fit"
        crs[key] = np.ones(self.ncrs)
        for i, cr in enumerate(crs):
            cr[key] = i
            self.assertEqual(cr[key], i)

    @unittest.skipIf(os.path.isfile("/.dockerenv"), "Plotting in Docker environment is not possible!")
    def test_16_plotting(self):
        ncrs = 1000
        crs = CosmicRaysBase(ncrs)
        crs['pixel'] = np.random.randint(0, 49152, ncrs)
        crs['log10e'] = 17. + 2.5 * np.random.random(ncrs)

        fname = "/tmp/energy_spectrum-%s.png" % user
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
        energies = np.random.uniform(18, 20, self.ncrs)
        ids = np.arange(self.ncrs)
        crs_arr = np.empty((self.ncrs,), dtype=np.dtype([("energy", "f"), ("cr_id", "i")]))
        crs_arr["energy"] = energies
        crs_arr["cr_id"] = ids
        crs = CosmicRaysBase(crs_arr)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs.get("energy") >= 18))
        self.assertTrue(np.all(crs.get("cr_id") == np.arange(self.ncrs)))
        self.assertTrue("energy" in crs.cosmic_rays.dtype.names)
        self.assertTrue("cr_id" in crs.cosmic_rays.dtype.names)
        # noinspection PyTypeChecker,PyUnresolvedReferences
        self.assertTrue(np.all(crs.energy() >= 18))

    def test_18_combine_keys(self):
        crs = CosmicRaysBase(self.ncrs)
        self.assertTrue(len(crs.keys()) == 0)
        crs['array'] = np.random.randint(0, 49152, self.ncrs)
        crs['ndarray'] = np.random.random((5, 2))
        crs['float'] = 5
        self.assertTrue('ndarray' in crs.keys())
        self.assertTrue('array' in crs.keys())
        self.assertTrue('float' in crs.keys())

    def test_19_keys_available(self):
        crs = CosmicRaysBase(self.ncrs)
        crs['array'] = np.random.randint(0, 49152, self.ncrs)
        crs['ndarray'] = np.random.random((5, 2))
        self.assertTrue('array' in crs.keys())
        self.assertTrue('ndarray' in crs.keys())

    def test_20_enumerate(self):
        crs = CosmicRaysBase(cosmic_rays=self.ncrs)
        imax1, imax2 = 0, 0
        for i, _ in enumerate(crs):
            imax1 = i + 1
        crs["energy"] = np.random.uniform(0, 1, self.ncrs)
        for i, _ in enumerate(crs):
            imax2 = i + 1
        self.assertTrue(imax1 == self.ncrs)
        self.assertTrue(imax2 == self.ncrs)

    def test_21_assign_with_list(self):
        crs_list = [1, 2, 3, 4]
        with self.assertRaises(NotImplementedError):
            CosmicRaysBase(crs_list)

    def test_22_numpy_integer(self):
        n = np.int16(64)
        crs = CosmicRaysBase(n)
        self.assertTrue(crs.ncrs == n)

    def test_23_access_non_existing_element(self):
        crs = CosmicRaysBase(self.ncrs)
        crs['array'] = np.random.randint(0, 49152, self.ncrs)
        with self.assertRaises(ValueError):
            crs["non_existing"]

    def test_24_set_unallowed_items(self):
        crs = CosmicRaysBase(self.ncrs)
        with self.assertRaises(KeyError):
            # case where the user does use the value as key and vice versa
            crs[[1, 2, 3]] = "key"

    def test_25_slicing_base(self):
        crs = CosmicRaysBase(self.ncrs)
        energy = np.random.random(self.ncrs)
        crs['energy'] = energy
        crs_sub = crs[energy < 0.3]
        self.assertTrue(hasattr(crs_sub, 'keys'))
        self.assertTrue(len(crs_sub) < self.ncrs)
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

    def test_27_save_readable(self):
        crs = CosmicRaysBase(self.ncrs)
        keys = ['a_str', 'a_float', 'an_int', 'ndarray', 'array', 'custom_array']
        entries = ['karl', 42.42, 4, np.random.random(size=(2, self.ncrs)),
                   np.random.random(self.ncrs), np.random.random(17)]
        for _key, _entry in zip(keys, entries):
            crs[_key] = _entry

        opath = '/tmp/cosmicraysbase-reeadable-%s.npz' % user
        crs.save_readable(opath)

        array = np.genfromtxt(opath)
        self.assertTrue(np.allclose(array, entries[keys.index('array')]))
        lines = "".join(open(opath).readlines())
        for _key, _entry in zip(keys, entries):
            self.assertTrue(_key in lines)


class TestCosmicRaysSets(unittest.TestCase):

    def setUp(self):
        self.ncrs = 10
        self.nsets = 15
        self.shape = (self.nsets, self.ncrs)

    def test_01_create(self):
        crsset = CosmicRaysSets(self.shape)
        self.assertEqual(crsset.shape, self.shape)
        self.assertEqual(crsset.ncrs, self.ncrs)
        self.assertEqual(crsset.nsets, self.nsets)

    def test_01a_create_with_None(self):
        crsset = CosmicRaysSets(self.shape)
        log10e = np.random.random(self.shape)
        crsset['log10e'] = log10e
        outpath = "/tmp/cosmicraysset-%s.npz" % user
        crsset.save(outpath)

        crsset2 = CosmicRaysSets(None)
        crsset2.load(outpath)
        self.assertTrue('log10e' in crsset2.keys())
        self.assertTrue((crsset2['log10e'] == log10e).all())
        os.remove(outpath)

    def test_01b_create_with_fake_object(self):
        class test:
            def __init__(self):
                self.type = "CosmicRaysSet"
        with self.assertRaises(AttributeError):
            t = test()
            CosmicRaysSets(t)

    def test_02_get_element_from_set(self):
        crsset = CosmicRaysSets(self.shape)
        # noinspection PyTypeChecker
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crsset["log10e"] == 0.))
        self.assertEqual(crsset["log10e"].shape, self.shape)

    def test_03_set_element(self):
        crsset = CosmicRaysSets(self.shape)
        energies = np.random.uniform(18, 20, size=self.shape)
        crsset["log10e"] = energies
        # noinspection PyTypeChecker
        self.assertTrue('log10e' in crsset.keys())
        self.assertTrue(len(crsset.log10e()) == self.nsets)
        self.assertTrue(np.all(crsset["log10e"] >= 18))

    def test_04_get_set_by_number(self):
        set_number = 3
        crsset = CosmicRaysSets(self.shape)
        crsset["creator"] = "Martin"
        subset = crsset[set_number]
        self.assertTrue(len(subset), self.ncrs)
        self.assertTrue(subset["creator"], "Martin")
        self.assertTrue(len(subset.cosmic_rays), self.ncrs)

    # noinspection PyTypeChecker
    def test_05_set_in_subset(self):
        set_number = 3
        crsset = CosmicRaysSets(self.shape)
        crsset["creator"] = "Martin"
        crsset["log10e"] = np.zeros(shape=self.shape)
        subset = crsset[set_number]
        subset["log10e"] = np.random.uniform(18, 20, self.ncrs)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(subset["log10e"] >= 18))
        idx_begin = int(self.ncrs * set_number)
        idx_end = int(self.ncrs * (set_number + 1))
        self.assertTrue(np.all(crsset.cosmic_rays[idx_begin:idx_end]["log10e"] >= 18))
        self.assertTrue(np.all(crsset.cosmic_rays[0:idx_begin]["log10e"] == 0))
        self.assertTrue(np.all(crsset.cosmic_rays[idx_end:]["log10e"] == 0))

    def test_06_copy(self):
        crs = CosmicRaysSets(self.shape)
        key = "an_int"
        crs[key] = 10
        crs2 = CosmicRaysSets(crs)
        crs[key] = -2
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crs2[key] == 10))

    def test_07_copy_array(self):
        crs = CosmicRaysSets(self.shape)
        key = "an_array"
        array = np.random.random(self.shape)
        crs[key] = array
        crs2 = CosmicRaysSets(crs)
        crs[key] = np.random.random(self.shape)
        # noinspection PyTypeChecker
        self.assertTrue((key in crs.keys()) and (key in crs2.keys()))
        self.assertTrue(key not in crs.general_object_store.keys())
        self.assertTrue(key not in crs2.general_object_store.keys())
        self.assertTrue(np.allclose(array, crs2[key]))

    def test_08_copy_gos_array(self):
        # gos = general object store
        crs = CosmicRaysSets(self.shape)
        key = "an_array"
        array = np.random.random(self.ncrs)
        crs[key] = array
        crs2 = CosmicRaysSets(crs)
        crs[key] = np.random.random(self.ncrs)
        # noinspection PyTypeChecker
        self.assertTrue((key in crs.keys()) and (key in crs2.keys()))
        self.assertTrue(key in crs.general_object_store.keys())
        self.assertTrue(key in crs2.general_object_store.keys())
        self.assertTrue(np.allclose(array, crs2[key]))

    def test_09_save(self):
        outpath = "/tmp/cosmicraysset-%s.pkl" % user
        crsset = CosmicRaysSets(self.shape)
        crsset["creator"] = "Marcus"
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        crsset.save(outpath)
        self.assertTrue(os.path.exists(outpath))
        os.remove(outpath)

    def test_09a_save_load(self):
        outpath = "/tmp/cosmicraysset_load1-%s.npz" % user
        crsset = CosmicRaysSets(self.shape)
        crsset["creator"] = "Marcus"
        crsset["log10e"] = np.zeros(shape=crsset.shape)
        crsset.save(outpath)

        crsset2 = CosmicRaysSets(self.shape)
        crsset2.load(outpath)
        os.remove(outpath)
        self.assertTrue("creator" in crsset2.keys())
        self.assertTrue("log10e" in crsset2.keys())
        self.assertTrue(np.allclose(crsset["log10e"], crsset2["log10e"]))

    def test_09b_save_load(self):
        outpath = "/tmp/cosmicraysset_load2-%s.npz" % user
        crsset = CosmicRaysSets(self.shape)
        crsset["creator"] = "Marcus"
        crsset.save(outpath)

        crsset2 = CosmicRaysSets(outpath)
        os.remove(outpath)
        self.assertTrue("creator" in crsset2.keys())

    def test_10_create_from_filename(self):
        # Create first the set and save it to file
        outpath = "/tmp/cosmicraysset-%s.npz" % user
        crsset = CosmicRaysSets(self.shape)
        crsset["creator"] = "Martin"
        crsset["log10e"] = np.ones(shape=crsset.shape)
        crsset.save(outpath)
        # reload the set as a new cosmic rays set
        crsset2 = CosmicRaysSets(outpath)
        os.remove(outpath)
        self.assertTrue(crsset2["creator"] == "Martin")
        self.assertTrue(np.shape(crsset2["log10e"]) == self.shape)
        # noinspection PyTypeChecker
        self.assertTrue(np.all(crsset2["log10e"] == 1))

    @unittest.skipIf(os.path.isfile("/.dockerenv"), "Plotting in Docker environment is not possible!")
    def test_11_plot(self):
        ncrs = 100
        crs = CosmicRaysSets((self.nsets, ncrs))
        crs['pixel'] = np.random.randint(0, 49152, (self.nsets, ncrs))
        crs['log10e'] = 18. + 2.5 * np.random.random((self.nsets, ncrs))

        crs.plot_eventmap()
        crs.plot_energy_spectrum()
        crs.plot_healpy_map()
        self.assertTrue(True)

    @unittest.skipIf(os.path.isfile("/.dockerenv"), "Plotting in Docker environment is not possible!")
    def test_12_plot_from_loaded_cosmic_rays_set(self):
        ncrs = 100
        crs = CosmicRaysSets((self.nsets, ncrs))
        crs['pixel'] = np.random.randint(0, 49152, (self.nsets, ncrs))
        crs['log10e'] = 18. + 2.5 * np.random.random((self.nsets, ncrs))
        fname = "/tmp/test_08-%s.npy" % user
        crs.save(fname)

        crs3 = CosmicRaysSets(fname)
        crs3.plot_eventmap(opath=fname.replace('.npy', '.png'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(fname.replace('.npy', '.png')))
        os.remove(fname)

    def test_13_keys_available(self):
        ncrs = 100
        crs = CosmicRaysSets((self.nsets, ncrs))
        crs['ndarray'] = np.random.randint(0, 49152, (self.nsets, ncrs))
        crs['array'] = np.random.random(self.nsets)
        crs['float'] = 5.
        self.assertTrue('ndarray' in crs.keys())
        self.assertTrue('array' in crs.keys())
        self.assertTrue('float' in crs.keys())

    def test_14_mask_subset(self):
        nsets = 100
        ndarray = np.random.randint(0, 49152, (nsets, self.ncrs))
        array = np.random.random(nsets)
        crs = CosmicRaysSets((nsets, self.ncrs))
        crs['ndarray'] = ndarray
        crs['array'] = array
        crs['string'] = 'blubb'
        crs['integer'] = 5
        mask = np.linspace(0, 1, nsets) < 0.3
        nsets_subset = 30
        crs_subset = crs[mask]
        self.assertTrue(nsets == crs.nsets)
        self.assertTrue((crs_subset.nsets == nsets_subset) and (crs_subset.ncrs == self.ncrs))
        self.assertTrue('ndarray' in crs_subset.keys())
        self.assertTrue('array' in crs_subset.keys())
        self.assertTrue(np.allclose(crs_subset['array'], array[mask]))
        self.assertEqual(crs_subset['string'], crs['string'])
        self.assertEqual(crs_subset['string'], 'blubb')
        self.assertEqual(crs_subset['integer'], crs['integer'])

    def test_15_indexing_subset(self):
        ndarray = np.random.randint(0, 49152, self.shape)
        array = np.random.random(self.nsets)
        crs = CosmicRaysSets(self.shape)
        crs['ndarray'] = ndarray
        crs['array'] = array
        crs['string'] = 'blubb'
        crs['integer'] = 5
        indexing = np.random.choice(np.arange(self.nsets), 10, replace=False)
        crs_subset = crs[indexing]
        self.assertTrue(self.nsets == crs.nsets)
        self.assertTrue('ndarray' in crs_subset.keys())
        self.assertTrue('array' in crs_subset.keys())
        self.assertTrue(crs_subset.nsets == 10)
        self.assertTrue(np.allclose(crs_subset['array'], array[indexing]))
        self.assertEqual(crs_subset['string'], crs['string'])
        self.assertEqual(crs_subset['string'], 'blubb')
        self.assertEqual(crs_subset['integer'], crs['integer'])

    def test_16_slicing_subset(self):
        ndarray = np.random.randint(0, 49152, self.shape)
        array = np.random.random(self.nsets)
        crs = CosmicRaysSets(self.shape)
        crs['ndarray'] = ndarray
        crs['array'] = array
        crs['string'] = 'blubb'
        crs['integer'] = 5
        low, up = 2, 10
        crs_subset = crs[low:up]
        self.assertTrue(self.nsets == crs.nsets)
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

    def test_17_access_functions(self):
        crs = CosmicRaysSets(self.shape)
        self.assertTrue(crs.shape, self.shape)
        crs["log10e"] = np.random.random(self.shape)
        self.assertTrue(crs["log10e"].shape, self.shape)
        self.assertTrue(crs.log10e().shape, self.shape)
        self.assertTrue(crs.shape, self.shape)
        creator = "Peter"
        crs["creator"] = creator
        self.assertTrue(crs["creator"], creator)
        self.assertTrue(crs.creator(), creator)
        self.assertTrue(crs.shape, self.shape)
        with self.assertRaises(TypeError):
            crs.shape()

    def test_18_failing_get(self):
        creator = "Peter"
        crs = CosmicRaysSets(self.shape)
        crs["creator"] = creator
        with self.assertRaises((ValueError, TypeError)):
            crs[["test"]]

    def test_19_create_from_crs_list(self):
        crs1 = CosmicRaysBase(self.ncrs)
        crs1["log10e"] = np.random.rand(self.ncrs)
        crs1["name"] = "set1"
        crs2 = CosmicRaysBase(self.ncrs)
        crs2["log10e"] = np.random.rand(self.ncrs)
        crs2["name"] = "set2"
        crs_s = [crs1, crs2]
        crs_set = CosmicRaysSets(crs_s)
        self.assertTrue(crs_set.shape == (2, self.ncrs))
        self.assertTrue(np.all(crs_set[0]["log10e"] == crs1["log10e"]))
        self.assertTrue(np.all(crs_set[1]["log10e"] == crs2["log10e"]))
        self.assertTrue(len(crs_set["name"]) == 2)
        self.assertTrue(np.all([crs_set["name"][i] == "set%i" % (i+1) for i in range(2)]))

    def test_20a_create_from_crs_non_cosmic_rays_list(self):
        log10e1 = np.random.rand(self.ncrs)
        log10e2 = np.random.rand(self.ncrs)
        crs_s = [log10e1, log10e2]
        with self.assertRaises(TypeError):
            CosmicRaysSets(crs_s)

    def test_20b_create_from_crs_list_unequal_nr_crs(self):
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

    def test_20c_create_from_crs_list_unequal_attributes(self):
        crs1 = CosmicRaysBase(self.ncrs)
        crs1["log10e"] = np.random.rand(self.ncrs)
        crs1["name"] = "set1"
        crs2 = CosmicRaysBase(self.ncrs)
        crs2["log10e"] = np.random.rand(self.ncrs)
        crs2["name"] = "set2"
        crs1["not_in_1"] = "test"
        crs_s = [crs1, crs2]
        with self.assertRaises(AttributeError):
            CosmicRaysSets(crs_s)

    def test_20d_create_from_crs_list_fake_crs(self):
        class Fake:
            def __init__(self):
                self.type = "CosmicRaysSet"

            def __len__(self):
                return 10

        crs1 = CosmicRaysBase(self.ncrs)
        crs1["log10e"] = np.random.rand(self.ncrs)
        crs2 = Fake()
        with self.assertRaises(TypeError):
            CosmicRaysSets([crs1, crs2])

        crs3 = Fake()
        with self.assertRaises(TypeError):
            CosmicRaysSets([crs2, crs3])

    def test_21_access_non_existing_object(self):
        crs = CosmicRaysSets(self.shape)
        crs['ndarray'] = np.random.randint(0, 49152, self.shape)
        with self.assertRaises(ValueError):
            crs["test"]

    def test_22_mask_ncrs(self):
        # if one dimensional mask, the slicing must be in the nsets dimension
        nsets = 1
        crs = CosmicRaysSets((nsets, self.ncrs))
        mask = np.ones(self.ncrs, dtype=bool)
        mask[0] = False
        with self.assertRaises(AssertionError):
            crs = crs[mask]

    def test_23_mask_nsets_ncrs(self):
        nsets, ncrs = 5, 100
        crs = CosmicRaysSets((nsets, ncrs))
        energies = np.linspace(0, 100, ncrs)
        crs['energy'] = energies
        mask = np.zeros((nsets, ncrs), dtype=bool)
        mask[:, crs['energy'] > 30] = True
        crs = crs[mask]
        self.assertTrue(crs.shape == (nsets, 70))
        self.assertTrue(crs.ncrs == 70)

    def test_24_mask_arbitrary(self):
        crs = CosmicRaysSets(self.shape)
        energy = np.random.random(self.shape)
        _id = np.arange(self.nsets)
        crs['energy'] = energy
        crs['id'] = _id
        crs['foo'] = 'foo'

        mask = np.zeros(self.shape, dtype=bool)
        nsets_sub, ncrs_sub = 3, 8
        mask[0:nsets_sub, 0:ncrs_sub] = True
        crs_sliced = crs[mask]
        self.assertTrue(crs_sliced.shape == (nsets_sub, ncrs_sub))
        self.assertTrue((crs_sliced.nsets == nsets_sub) & (crs_sliced.ncrs == ncrs_sub))

        keys = crs_sliced.keys()
        self.assertTrue(('energy' in keys) & ('id' in keys) & ('foo' in keys))
        self.assertTrue(np.array_equal(crs_sliced['energy'], energy[0:nsets_sub, 0:ncrs_sub]))
        self.assertTrue(np.array_equal(crs_sliced['id'], _id[0:nsets_sub]))
        self.assertTrue(crs_sliced['foo'] == 'foo')
        # check that old instance is not affected
        self.assertTrue(crs.shape == self.shape)
        self.assertTrue((crs.nsets == self.nsets) & (crs.ncrs == self.ncrs))
        self.assertTrue(np.array_equal(crs['energy'], energy))
        self.assertTrue(np.array_equal(crs['id'], _id))
        self.assertTrue(crs['foo'] == 'foo')

        # arbitrary masks can not be applied
        mask = np.random.randint(0, 2, size=self.shape).astype(bool)
        with self.assertRaises(AssertionError):
            crs_sliced = crs[mask]

    def test_25_save_readable(self):
        crs = CosmicRaysSets(self.shape)
        keys = ['a_str', 'a_float', 'an_int', 'ndarray', 'array', 'custom_array']
        entries = ['karl', 42.42, 4, np.random.random(size=self.shape),
                   np.random.random(self.nsets), np.random.random(self.ncrs)]
        for _key, _entry in zip(keys, entries):
            crs[_key] = _entry

        opath = '/tmp/cosmicrayssets-reeadable-%s.npz' % user
        crs.save_readable(opath)

        lines = "".join(open(opath).readlines())
        for _key, _entry in zip(keys, entries):
            self.assertTrue(_key in lines)


if __name__ == '__main__':
    unittest.main()
