import unittest
import os
import numpy as np

from astrotools import coord, gamale, healpytools as hpt
from astrotools.simulations import ObservedBound

__author__ = 'Marcus Wirtz'

nside = 64
ncrs = 1000
nsets = 10
np.random.seed(0)


class TestObservedBound(unittest.TestCase):

    def setUp(self):
        self.nside = 64
        self.nsets = 10
        self.ncrs = 1000
        self.shape = (self.nsets, self.ncrs)

    def test_01_n_cosmic_rays(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        self.assertEqual(sim.ncrs, self.ncrs)

    def test_02_nsets(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        self.assertEqual(sim.nsets, self.nsets)

    def test_03_keyword_setup(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
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
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        log10e = np.random.random(size=self.shape)
        charge = np.random.randint(0, 10, size=self.shape)
        sim.set_energy(log10e_min=log10e)
        sim.set_charges(charge=charge)
        crs = sim.get_data()
        self.assertTrue(np.allclose(crs['log10e'], log10e) and np.allclose(crs['charge'], charge))

        sim2 = ObservedBound(self.nside, self.nsets, self.ncrs)
        log10e = np.random.random(self.ncrs)
        charge = np.random.random(self.ncrs)
        sim2.set_energy(log10e)
        sim2.set_charges(charge)
        self.assertTrue(np.allclose(sim2.crs['log10e'], log10e) and np.allclose(sim2.crs['charge'], charge))

        sim3 = ObservedBound(self.nside, self.nsets, self.ncrs)
        log10e = np.random.random(self.nsets)
        charge = np.random.random(self.nsets)
        with self.assertRaises(Exception):
            sim3.set_energy(log10e)
            sim3.set_charges(log10e)

    def test_05_set_n_random_sources(self):
        n = 5
        fluxes = np.random.random(n)
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim.set_sources(n, fluxes=fluxes)
        self.assertTrue(sim.sources.shape[1] == n)
        self.assertTrue(np.allclose(fluxes, sim.source_fluxes))

    def test_06_set_n_sources(self):
        v_src = np.random.rand(30).reshape((3, 10))
        fluxes = np.random.random(10)
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim.set_sources(v_src, fluxes=fluxes)
        self.assertTrue(np.allclose(v_src, sim.sources))
        self.assertTrue(np.allclose(fluxes, sim.source_fluxes))

    def test_07_smear_sources_dynamically(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
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
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim.arrival_setup(0.)
        crs = sim.get_data(convert_all=True)
        x = np.abs(np.mean(crs['x']))
        y = np.abs(np.mean(crs['y']))
        z = np.abs(np.mean(crs['z']))
        self.assertTrue((x < 0.03) & (y < 0.03) & (z < 0.03))

    def test_09_exposure(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim.apply_exposure()
        sim.arrival_setup(0.)
        crs = sim.get_data(convert_all=True)
        vecs_eq = coord.gal2eq(coord.ang2vec(np.hstack(crs['lon']), np.hstack(crs['lat'])))
        lon_eq, lat_eq = coord.vec2ang(vecs_eq)
        self.assertTrue(np.abs(np.mean(lon_eq)) < 0.05)
        self.assertTrue((np.mean(lat_eq) < -0.5) & (np.mean(lat_eq) > - 0.55))

    def test_10_charge(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        charge = 2
        sim.set_charges(charge)
        self.assertTrue(sim.crs['charge'] == charge)

    def test_11_xmax_setup(self):
        sim1 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim1.set_energy(19.)
        sim1.set_charges(2)
        sim1.set_xmax('stable')
        check1 = (sim1.crs['xmax'] > 500) & (sim1.crs['xmax'] < 1200)

        sim2 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim2.set_energy(19.)
        sim2.set_charges(2)
        sim2.set_xmax('stable')
        check2 = (sim2.crs['xmax'] > 500) & (sim2.crs['xmax'] < 1200)

        sim3 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim3.set_energy(19.)
        sim3.set_charges(2)
        sim3.set_xmax('empiric')
        check3 = (sim3.crs['xmax'] > 500) & (sim3.crs['xmax'] < 1200)

        sim4 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim4.set_energy(19.)
        sim4.set_charges(2)
        sim4.set_xmax('double')
        check4 = (sim4.crs['xmax'] > 500) & (sim4.crs['xmax'] < 1200)

        self.assertTrue(check1.all() and check2.all() and check3.all() and check4.all())

    def test_12_xmax_mass(self):
        sim1 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim2 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim1.set_energy(19.)
        sim2.set_energy(19.)
        sim1.set_charges('equal')
        sim2.set_charges(26)
        sim1.set_xmax('double')
        sim2.set_xmax('double')
        # Xmax of iron should be smaller (interact higher in atmosphere)
        self.assertTrue(np.mean(sim1.crs['xmax']) > np.mean(sim2.crs['xmax']))

    def test_13_xmax_energy(self):
        sim1 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim2 = ObservedBound(self.nside, self.nsets, self.ncrs)
        sim1.set_energy(20. * np.ones(self.shape))
        sim2.set_energy(19. * np.ones(self.shape))
        sim1.set_charges(1)
        sim2.set_charges(1)
        sim1.set_xmax('double')
        sim2.set_xmax('double')
        # Xmax for higher energy is bigger
        self.assertTrue(np.mean(sim1.crs['xmax']) > np.mean(sim2.crs['xmax']))

    def test_14_lensing_map(self):
        lens_path = os.path.dirname(os.path.realpath(__file__)) + '/toy-lens/jf12-regular.cfg'
        toy_lens = gamale.Lens(lens_path)
        nside = toy_lens.nside
        sim = ObservedBound(nside, self.nsets, self.ncrs)
        sim.set_energy(19.*np.ones(self.shape))
        sim.set_charges(1)
        sim.set_xmax('empiric')
        sim.set_sources('gamma_agn')
        sim.lensing_map(toy_lens)
        # Xmax for higher energy is bigger
        self.assertTrue(sim.lensed)
        self.assertTrue(np.shape(sim.cr_map) == (1, sim.npix))
        self.assertAlmostEqual(np.sum(sim.cr_map), 1.)
        self.assertTrue(np.min(sim.cr_map) < np.max(sim.cr_map))

    def test_15_exposure(self):
        nsets = 100
        sim = ObservedBound(self.nside, nsets, self.ncrs)
        sim.apply_exposure(a0=-35.25, zmax=60)
        sim.arrival_setup(0.2)
        crs = sim.get_data(convert_all=True)
        lon, lat = np.hstack(crs['lon']), np.hstack(crs['lat'])
        ra, dec = coord.vec2ang(coord.gal2eq(coord.ang2vec(lon, lat)))
        exp = coord.exposure_equatorial(dec, a0=-35.25, zmax=60)
        self.assertTrue((exp > 0).all())

    def test_16_high_nside(self):
        nside_high = 256
        sim = ObservedBound(nside_high, nsets=self.nsets, ncrs=self.ncrs)
        self.assertTrue(sim.crs['nside'] == nside_high)
        self.assertTrue(sim.crs.nside() == nside_high)
        sim.arrival_setup(1.)
        self.assertTrue(np.max(sim.crs['pixel']) >= 0.8 * sim.npix)

    def test_17_energy_rigidity_set(self):
        ncrs = 100
        e = 19.5
        sim = ObservedBound(self.nside, self.nsets, ncrs)
        sim.set_energy(e * np.ones((self.nsets, ncrs)))
        sim.set_charges(2)
        sim.set_rigidity_bins(np.linspace(17., 20.48, 175))
        self.assertTrue((sim.crs['log10e'] == e).all())

    def test_18_energy_spectra(self):
        nsets = 100
        sim = ObservedBound(self.nside, nsets, self.ncrs)
        log10e_power_3 = sim.set_energy(log10e_min=19., log10e_max=21., energy_spectrum='power_law', gamma=-3)
        ebin = 0.1
        for e in np.arange(19.1, 20.1, ebin):
            sum_low = np.sum((log10e_power_3 >= e-ebin) & (log10e_power_3 < e))
            sum_high = np.sum((log10e_power_3 >= e) & (log10e_power_3 < e+ebin))
            self.assertTrue(sum_low > sum_high)
        sim2 = ObservedBound(self.nside, nsets, self.ncrs)
        log10e_power_4 = sim2.set_energy(log10e_min=19., log10e_max=21., energy_spectrum='power_law', gamma=-4)
        # higher energies for flatter spectrum
        self.assertTrue(np.mean(log10e_power_3) > np.mean(log10e_power_4))
        sim3 = ObservedBound(self.nside, nsets, self.ncrs)
        log10e_auger_fit = sim3.set_energy(log10e_min=19., log10e_max=21., energy_spectrum='auger_fit')
        self.assertTrue(np.mean(log10e_power_3) > np.mean(log10e_auger_fit))
        self.assertTrue(np.mean(log10e_power_4) < np.mean(log10e_auger_fit))

    def test_19_apply_uncertainties(self):
        sim = ObservedBound(self.nside, self.nsets, self.ncrs)
        log10e = sim.set_energy(log10e_min=19., log10e_max=21., energy_spectrum='power_law', gamma=-3)
        sim.set_charges('mixed')
        xmax = sim.set_xmax()
        sim.set_sources(10)
        sim.set_rigidity_bins(np.arange(17., 20.5, 0.02))
        sim.smear_sources(delta=0.1, dynamic=True)
        sim.arrival_setup(1.)
        vecs = hpt.pix2vec(sim.nside, np.hstack(sim.crs['pixel']))
        sim.apply_uncertainties(err_e=0.1, err_a=1, err_xmax=10)
        # check that array are not equal but deviations are smaller than 5 sigma
        self.assertTrue(not (log10e == sim.crs['log10e']).all())
        self.assertTrue((np.abs(10**(log10e - 18) - 10**(sim.crs['log10e'] - 18)) < 5*0.1*10**(log10e - 18)).all())
        self.assertTrue(not (xmax == sim.crs['xmax']).all())
        self.assertTrue((np.abs(xmax - sim.crs['xmax']) < 50).all())
        vec_unc = coord.ang2vec(np.hstack(sim.crs['lon']), np.hstack(sim.crs['lat']))
        self.assertTrue(not (coord.angle(vecs, vec_unc) == 0).all())
        self.assertTrue((coord.angle(vecs, vec_unc) < np.deg2rad(10)).all())


if __name__ == '__main__':
    unittest.main()
