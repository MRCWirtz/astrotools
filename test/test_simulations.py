import unittest

import numpy as np
from astrotools.simulations import CosmicRaySimulation

__author__ = 'Marcus Wirtz'

nside = 64
ncrs = 1000
nsets = 10


class TestCosmicRaySimulation(unittest.TestCase):

    def test_01_n_cosmic_rays(self):
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        self.assertEqual(sim.ncrs, ncrs)

    def test_02_stat(self):
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        self.assertEqual(sim.nsets, nsets)

    def test_03_keyword_setup(self):
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        sim.set_energy(log10e_min=19.)
        sim.set_charges(charge='AUGER')
        sim.set_sources(sources='sbg')
        sim.smear_sources(delta=0.1)
        sim.apply_exposure()
        sim.arrival_setup(fsig=0.4)
        crs = sim.get_data(convert_all=True)
        self.assertEqual(crs['pixel'].shape, crs['lon'].shape, crs['log10e'].shape)

    def test_04_set_energy_charge_arrays(self):
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        log10e = np.random.rand(nsets * ncrs).reshape((nsets, ncrs))
        charge = np.random.randint(0, 10, nsets * ncrs).reshape((nsets, ncrs))
        sim.set_energy(log10e_min=log10e)
        sim.set_charges(charge=charge)
        crs = sim.get_data()
        self.assertTrue(np.allclose(crs['log10e'], log10e) and np.allclose(crs['charge'], charge))

    def test_05_set_n_random_sources(self):
        n = 5
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        sim.set_sources(n)
        self.assertTrue(sim.sources.shape[1] == n)

    def test_06_set_n_sources(self):
        v_src = np.random.rand(30).reshape((3, 10))
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        sim.set_sources(v_src)
        self.assertTrue(np.allclose(v_src, sim.sources))

    def test_07_smear_sources_dynamically(self):
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        sim.set_energy(log10e_min=19.)
        sim.set_charges('AUGER')
        sim.set_sources(5)
        sim.set_rigidity_bins(np.arange(17., 20.5, 0.02))
        sim.smear_sources(delta=0.1, dynamic=True)
        sim.arrival_setup(1.)
        self.assertTrue(True)

    def test_08_isotropy(self):
        sim = CosmicRaySimulation(nside, nsets, ncrs)
        sim.apply_exposure()
        sim.arrival_setup(0.)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
