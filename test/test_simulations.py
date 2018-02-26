import unittest
import numpy as np

from astrotools import coord
from astrotools.simulations import ObservedBound

__author__ = 'Marcus Wirtz'

nside = 64
ncrs = 1000
nsets = 10
np.random.seed(0)


class TestObservedBound(unittest.TestCase):

    def test_01_n_cosmic_rays(self):
        sim = ObservedBound(nside, nsets, ncrs)
        self.assertEqual(sim.ncrs, ncrs)

    def test_02_stat(self):
        sim = ObservedBound(nside, nsets, ncrs)
        self.assertEqual(sim.nsets, nsets)

    def test_03_keyword_setup(self):
        sim = ObservedBound(nside, nsets, ncrs)
        sim.set_energy(log10e_min=19.)
        sim.set_charges(charge='mixed')
        sim.set_xmax('double')
        sim.set_sources(sources='sbg')
        sim.smear_sources(delta=0.1)
        sim.apply_exposure()
        sim.arrival_setup(fsig=0.4)
        crs = sim.get_data(convert_all=True)
        self.assertEqual(crs['pixel'].shape, crs['lon'].shape, crs['log10e'].shape)

    def test_04_set_energy_charge_arrays(self):
        sim = ObservedBound(nside, nsets, ncrs)
        log10e = np.random.rand(nsets * ncrs).reshape((nsets, ncrs))
        charge = np.random.randint(0, 10, nsets * ncrs).reshape((nsets, ncrs))
        sim.set_energy(log10e_min=log10e)
        sim.set_charges(charge=charge)
        crs = sim.get_data()
        self.assertTrue(np.allclose(crs['log10e'], log10e) and np.allclose(crs['charge'], charge))

    def test_05_set_n_random_sources(self):
        n = 5
        fluxes = np.random.random(n)
        sim = ObservedBound(nside, nsets, ncrs)
        sim.set_sources(n, fluxes=fluxes)
        self.assertTrue(sim.sources.shape[1] == n)
        self.assertTrue(np.allclose(fluxes, sim.source_fluxes))

    def test_06_set_n_sources(self):
        v_src = np.random.rand(30).reshape((3, 10))
        fluxes = np.random.random(10)
        sim = ObservedBound(nside, nsets, ncrs)
        sim.set_sources(v_src, fluxes=fluxes)
        self.assertTrue(np.allclose(v_src, sim.sources))
        self.assertTrue(np.allclose(fluxes, sim.source_fluxes))

    def test_07_smear_sources_dynamically(self):
        sim = ObservedBound(nside, nsets, ncrs)
        sim.set_energy(log10e_min=19.)
        sim.set_charges('AUGER')
        sim.set_sources(1)
        sim.set_rigidity_bins(np.arange(17., 20.5, 0.02))
        sim.smear_sources(delta=0.1, dynamic=True)
        sim.arrival_setup(1.)
        crs = sim.get_data(convert_all=True)
        rigs = sim.rigidities
        rig_med = np.median(rigs)
        vecs1 = coord.ang2vec(crs['lon'][rigs >= rig_med], crs['lat'][rigs >= rig_med])
        vecs2 = coord.ang2vec(crs['lon'][rigs < rig_med], crs['lat'][rigs < rig_med])
        # Higher rigidities experience higher deflections
        self.assertTrue(np.mean(coord.angle(vecs1, sim.sources)) < np.mean(coord.angle(vecs2, sim.sources)))

    def test_08_isotropy(self):
        sim = ObservedBound(nside, nsets, ncrs)
        sim.arrival_setup(0.)
        crs = sim.get_data(convert_all=True)
        x = np.abs(np.mean(crs['x']))
        y = np.abs(np.mean(crs['y']))
        z = np.abs(np.mean(crs['z']))
        self.assertTrue((x < 0.03) & (y < 0.03) & (z < 0.03))

    def test_09_exposure(self):
        sim = ObservedBound(nside, nsets, ncrs)
        sim.apply_exposure()
        sim.arrival_setup(0.)
        crs = sim.get_data(convert_all=True)
        vecs_eq = coord.gal2eq(coord.ang2vec(np.hstack(crs['lon']), np.hstack(crs['lat'])))
        lon_eq, lat_eq = coord.vec2ang(vecs_eq)
        self.assertTrue(np.abs(np.mean(lon_eq)) < 0.05)
        self.assertTrue((np.mean(lat_eq) < -0.5) & (np.mean(lat_eq) > - 0.55))

    def test_10_charge(self):
        sim = ObservedBound(nside, nsets, ncrs)
        charge = 2
        sim.set_charges(charge)
        self.assertTrue(sim.crs['charge'] == charge)

    def test_11_xmax_setup(self):
        sim = ObservedBound(nside, nsets, ncrs)
        sim.set_energy(19.)
        sim.set_charges(2)
        sim.set_xmax('stable')
        sim.set_xmax('empiric')
        sim.set_xmax('double')
        check = (sim.crs['xmax'] > 500) & (sim.crs['xmax'] < 1200)
        self.assertTrue(check.all())

    def test_12_xmax_mass(self):
        sim1 = ObservedBound(nside, nsets, ncrs)
        sim2 = ObservedBound(nside, nsets, ncrs)
        sim1.set_energy(19.)
        sim2.set_energy(19.)
        sim1.set_charges(1)
        sim2.set_charges(26)
        sim1.set_xmax('double')
        sim2.set_xmax('double')
        # Xmax of iron should be smaller (interact higher in atmosphere)
        self.assertTrue(np.mean(sim1.crs['xmax']) > np.mean(sim2.crs['xmax']))

    def test_13_xmax_energy(self):
        sim1 = ObservedBound(nside, nsets, ncrs)
        sim2 = ObservedBound(nside, nsets, ncrs)
        sim1.set_energy(20. * np.ones((nsets, ncrs)))
        sim2.set_energy(19. * np.ones((nsets, ncrs)))
        sim1.set_charges(1)
        sim2.set_charges(1)
        sim1.set_xmax('double')
        sim2.set_xmax('double')
        # Xmax for higher energy is bigger
        self.assertTrue(np.mean(sim1.crs['xmax']) > np.mean(sim2.crs['xmax']))


if __name__ == '__main__':
    unittest.main()
