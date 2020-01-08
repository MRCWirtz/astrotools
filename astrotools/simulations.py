""" Module to setup a parametrized simulation, that work on probability distributions """
import os
import numpy as np
from astrotools import auger, coord, cosmic_rays, healpytools as hpt, nucleitools as nt

PATH = os.path.split(__file__)[0]


def set_fisher_smeared_sources(nside, sources, delta, source_fluxes=None):
    """
    Smears the source positions (optional fluxes) with a fisher distribution of width delta.

    :param nside: nside of the HEALPix pixelization (default: 64)
    :type nside: int
    :param sources: array of shape (3, n_sources) that point towards the center of the sources
    :param delta: float or array with same length as sources: width of the fisher distribution (in radians)
    :param source_fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sources).
    :return: healpy map (with npix entries) for the smeared sources normalized to sum 1
    """
    npix = hpt.nside2npix(nside)
    nsrc = np.shape(sources)[1]
    eg_map = np.zeros(npix)
    if isinstance(delta, (int, float)):
        delta = np.repeat(delta, nsrc)
    if len(delta) != nsrc:
        raise ValueError("Number of deltas must be 1 or equal to number of sources")
    for i, v_src in enumerate(sources.T):
        eg_map_add = hpt.fisher_pdf(nside, *v_src, k=1. / delta[i] ** 2)
        if source_fluxes is not None:
            eg_map_add *= source_fluxes[i]
        eg_map += eg_map_add
    return eg_map / eg_map.sum()


def sample_from_m_distributions(weight_matrix, n):
    """
    Sample n events from m distributions each having k bins and defined in weight_matrix.

    :param weight_matrix: shape (m, k), hosting m different distributions each with k bins.
    :param n: Number of events drawn for each of the m distributions.
    :return indices: Indice array with values 0..(k-1), in shape (m, n)
    """
    weight_matrix /= weight_matrix.sum(axis=1, keepdims=True)
    s, r = weight_matrix.cumsum(axis=1), np.random.random(n)
    return (r[np.newaxis, np.newaxis] > s[:, :, np.newaxis]).sum(axis=1)


class BaseSimulation:
    """
    Base class where the classes ObservedBound() and SourceBound() inherit from.
    """

    def __init__(self, nsets, ncrs):
        nsets = int(nsets)
        ncrs = int(ncrs)
        self.shape = (nsets, ncrs)
        self.crs = cosmic_rays.CosmicRaysSets((nsets, ncrs))
        self.nsets = nsets
        self.ncrs = ncrs
        self.signal_label = None

    def set_xmax(self, z2a='double', model='EPOS-LHC'):
        """
        Calculate Xmax bei gumbel distribution for the simulated energies and charges.

        :param z2a: How the charge is converted to mass number ['double', 'empiric', 'stable', 'abundance']
        :param model: Hadronic interaction for gumbel distribution
        :return: no return
        """
        assert 'xmax' not in self.crs.keys(), "Try to re-assign xmax values!"

        if (not hasattr(self.crs, 'log10e')) or (not hasattr(self.crs, 'charge')):
            raise Exception("""Use function set_energy() and set_charges() before using function set_xmax.
                            If you Use SourceBound simulation attenuate() also has to be used additionally""")
        mass = getattr(nt.Charge2Mass(self.crs['charge']), z2a)()
        mass = np.hstack(mass) if isinstance(mass, np.ndarray) else mass
        xmax = auger.rand_xmax(np.hstack(self.crs['log10e']), mass, model=model)
        self.crs['xmax'] = np.reshape(xmax, self.shape)

        return self.crs['xmax']

    def apply_uncertainties(self, err_e=None, err_a=None, err_xmax=None):
        """
        Apply measurement uncertainties.

        :param err_e: relative uncertainty on the energy (typical 0.14)
        :param err_a: angular uncertainty in degree on the arrival directions (typical 0.5 degree)
        :param err_xmax: absolute uncertainty on the shower depth xmax (typical 15 g/cm^2)
        """
        if err_e is not None:
            self.crs['log10e'] += np.log10(1 + np.random.normal(scale=err_e, size=self.shape))

        if err_a is not None:
            vecs = coord.rand_fisher_vec(self.crs['vecs'], 1./np.deg2rad(err_a)**2)
            lon, lat = coord.vec2ang(vecs)
            self.crs['lon'] = lon.reshape(self.shape)
            self.crs['lat'] = lat.reshape(self.shape)

        if err_xmax is not None:
            self.crs['xmax'] += np.random.normal(err_xmax)

    def shuffle_events(self):
        """
        Independently shuffle the cosmic rays of each set.
        """
        # This function can be simplified in the future using np.take_along_axis()
        shuffle_ids = np.random.permutation(np.prod(self.shape)).reshape(self.shape)
        shuffle_ids = np.argsort(shuffle_ids, axis=1)
        sets_ids = np.repeat(np.arange(self.nsets), self.ncrs).reshape(self.shape)
        for _key in self.crs.shape_array.dtype.names:
            self.crs[_key] = self.crs[_key][sets_ids, shuffle_ids]
        sets_ids_3d = np.repeat(np.arange(3), np.prod(self.shape)).reshape((3,) + self.shape)
        self.crs['vecs'] = self.crs['vecs'][sets_ids_3d, sets_ids, np.stack([shuffle_ids] * 3)]
        self.signal_label = self.signal_label[sets_ids, shuffle_ids]


class ObservedBound(BaseSimulation):
    """
    Class to simulate cosmic ray arrival scenario by sources located at the sky, including energies, charges, smearing
    and galactic magnetic field effects.
    This is an observed bound simulation, thus energies and composition is set on Earth and might differ at sources.
    """

    def __init__(self, nside, nsets, ncrs):
        """
        Initialization of ObservedBound simulation.

        :param nside: nside of the HEALPix pixelization (default: 64)
        :param nsets: number of simulated cosmic ray sets
        :param ncrs: number of cosmic rays per set
        """
        BaseSimulation.__init__(self, nsets, ncrs)
        self.nside = nside
        self.npix = hpt.nside2npix(nside)

        self.crs['nside'] = nside
        self.sources = None
        self.source_fluxes = None

        self.rigidities = None
        self.rig_bins = None
        self.cr_map = None
        self.lensed = None
        self.exposure = {'map': None, 'a0': None, 'zmax': None}
        self.signal_idx = None

    def set_energy(self, log10e_min, log10e_max=None, energy_spectrum=None, **kwargs):
        """
        Setting the energies of the simulated cosmic ray set.

        :param log10e_min: Either minimum energy (in log10e) for AUGER setup or power-law
                           or numpy.array of energies in shape (nsets, ncrs)
        :type log10e_min: Union[np.ndarray, float]
        :param log10e_max: Maximum energy for AUGER setup
        :param energy_spectrum: model that is defined in below class EnergySpectrum
        :param kwargs: additional named keywords which will be passed to class EnergySpectrum
        :return: energies in log10e
        """
        assert 'log10e' not in self.crs.keys(), "Try to re-assign energies!"

        if isinstance(log10e_min, np.ndarray):
            if log10e_min.shape == self.shape:
                self.crs['log10e'] = log10e_min
            elif log10e_min.size == self.ncrs:
                print("Warning: the same energies have been used for all simulated sets (nsets).")
                self.crs['log10e'] = np.tile(log10e_min, self.nsets).reshape(self.shape)
            else:
                raise Exception("Shape of input energies not in format (nsets, ncrs).")
        elif isinstance(log10e_min, (float, np.float, int, np.int)):
            if energy_spectrum is not None:
                log10e = getattr(EnergySpectrum(self.shape, log10e_min, log10e_max), energy_spectrum)(**kwargs)
            else:
                if (log10e_min < 17.) or (log10e_min > 21.):
                    print("Warning: Specified parameter log10e_min below 17 or above 20.5.")
                log10e = auger.rand_energy_from_auger(self.shape, log10e_min=log10e_min, log10e_max=log10e_max)
            self.crs['log10e'] = log10e
        else:
            raise Exception("Input of emin could not be understood.")

        return self.crs['log10e']

    def set_charges(self, charge, **kwargs):
        """
        Setting the charges of the simulated cosmic ray set.

        :param charge: Either charge number that is used or numpy.array of charges in shape (nsets, ncrs) or keyword
        :type: charge: Union[np.ndarray, str, float]
        :return: charges
        """
        assert 'charge' not in self.crs.keys(), "Try to re-assign charges!"

        if isinstance(charge, np.ndarray):
            if charge.shape == self.shape:
                self.crs['charge'] = charge
            elif charge.size == self.ncrs:
                self.crs['charge'] = np.tile(charge, self.nsets).reshape(self.shape)
            else:
                raise Exception("Shape of input charges not in format (nsets, ncrs).")
        elif isinstance(charge, (float, np.float, int, np.int)):
            self.crs['charge'] = charge
        elif isinstance(charge, str):
            if not hasattr(self.crs, 'log10e'):
                raise Exception("Use function set_energy() before accessing a composition model.")
            self.crs['charge'] = getattr(CompositionModel(self.shape, self.crs['log10e']), charge.lower())(**kwargs)
        elif isinstance(charge, dict):
            z = np.array([nt.ELEMENT_CHARGE[key] for key in charge])
            self.crs['charge'] = np.random.choice(z, self.shape, p=[charge[key] for key in charge])
        else:
            raise Exception("Input of charge could not be understood.")

        return self.crs['charge']

    def set_xmax(self, z2a='double', model='EPOS-LHC'):
        """
        Calculate Xmax bei gumbel distribution for the simulated energies and charges.
        For more information refer to BaseSimulation.set_xmax().
        """
        return BaseSimulation.set_xmax(self, z2a, model)

    def set_sources(self, sources, fluxes=None):
        """
        Define source position and optional weights (cosmic ray luminosity).

        :param sources: array of shape (3, n_sources) that point towards the center of the sources or integer for
                        number of random sources or keyword ['sbg']
        :param fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sources).
        :return: no return
        """
        if isinstance(sources, np.ndarray):
            if (len(np.shape(sources)) == 1) and len(sources) == 3:
                sources = np.reshape(sources, (3, 1))
            assert len(np.shape(sources)) == 2
            assert np.shape(sources)[0] == 3
            self.sources = sources
            if fluxes is not None:
                assert fluxes.size == len(sources.T)
                self.source_fluxes = fluxes
        elif isinstance(sources, (int, np.int)):
            src_pix = np.random.randint(0, self.npix, sources)
            self.sources = np.array(hpt.pix2vec(self.nside, src_pix))
            if fluxes is not None:
                assert fluxes.size == sources
                self.source_fluxes = fluxes
        elif isinstance(sources, str):
            self.sources, self.source_fluxes = getattr(SourceScenario(), sources.lower())()[:2]
        else:
            raise Exception("Source scenario not understood.")

    def set_rigidity_bins(self, lens_or_bins, cover_rigidity=True):
        """
        Defines the bin centers of the rigidities.

        :param lens_or_bins: Either the binning of the lens (left bin borders) or the lens itself
        :return: no return
        """
        if self.rig_bins is None:
            if 'log10e' not in self.crs.keys():
                raise Exception("Cannot define rigidity bins without energies specified.")
            if 'charge' not in self.crs.keys():
                print("Warning: Energy dependent deflection instead of rigidity dependent (set_charges to avoid)")

            if isinstance(lens_or_bins, np.ndarray):
                bins = lens_or_bins  # type: np.array
                dbins = bins[1] - bins[0]
            else:
                bins = np.array(lens_or_bins.log10r_mins)
                dbins = lens_or_bins.dlog10r
            rigidities = self.crs['log10e']
            if 'charge' in self.crs.keys():
                rigidities = rigidities - np.log10(self.crs['charge'])
            if cover_rigidity:
                assert np.all(np.min(rigidities) >= np.min(bins - dbins / 2.)), "Rigidities not covered by lens!"
            idx = np.digitize(rigidities, bins) - 1
            rigs = (bins + dbins / 2.)[idx]
            rigs = rigs.reshape(self.shape)
            self.rigidities = rigs
            self.rig_bins = np.unique(rigs)

        return self.rig_bins

    def smear_sources(self, delta, dynamic=None):
        """
        Smears the source positions with a fisher distribution of width delta (optional dynamic smearing).

        :param delta: either the constant width of the fisher distribution or dynamic (delta / R[10 EV]), in radians
        :param dynamic: if True, function applies dynamic smearing (delta / R[EV]) with value delta at 10 EV rigidity
        :return: no return
        """
        if self.sources is None:
            raise Exception("Cannot smear sources without positions.")

        if (dynamic is None) or (dynamic is False):
            shape = (1, self.npix)
            eg_map = np.reshape(set_fisher_smeared_sources(self.nside, self.sources, delta, self.source_fluxes), shape)
        else:
            if self.rig_bins is None:
                raise Exception("Cannot dynamically smear sources without rigidity bins (use set_rigidity_bins()).")
            eg_map = np.zeros((self.rig_bins.size, self.npix))
            for i, rig in enumerate(self.rig_bins):
                delta_temp = delta / 10 ** (rig - 19.)
                eg_map[i] = set_fisher_smeared_sources(self.nside, self.sources, delta_temp, self.source_fluxes)
        self.cr_map = eg_map

    def lensing_map(self, lens, cache=None):
        """
        Apply a galactic magnetic field to the extragalactic map.

        :param lens: Instance of astrotools.gamale.Lens class, for the galactic magnetic field
        :param cache: Caches all the loaded lens parts (increases speed, but may consume a lot of memory!)
        :return: no return
        """
        if self.lensed:
            print("Warning: Cosmic Ray maps were already lensed before.")

        if self.rig_bins is None:
            self.set_rigidity_bins(lens)

        if self.cr_map is None:
            self._fix_point_source()

        arrival_map = np.zeros((self.rig_bins.size, self.npix))
        for i, rig in enumerate(self.rig_bins):
            lp = lens.get_lens_part(rig, cache=cache)
            eg_map_bin = self.cr_map[0] if self.cr_map.size == self.npix else self.cr_map[i]
            lensed_map = lp.dot(eg_map_bin)
            if not cache:
                del lp.data, lp.indptr, lp.indices
            arrival_map[i] = lensed_map / np.sum(lensed_map) if np.sum(lensed_map) > 0 else 1. / self.npix

        self.lensed = True
        self.cr_map = arrival_map

    def apply_exposure(self, a0=-35.25, zmax=60):
        """
        Apply the exposure (coverage) of any experiment (default: AUGER) to the observed maps.

        :param a0: equatorial declination [deg] of the experiment (default: AUGER, a0=-35.25 deg)
        :type a0: float, int
        :param zmax: maximum zenith angle [deg] for the events
        :return: no return
        """
        self.exposure.update({'map': hpt.exposure_pdf(self.nside, a0, zmax), 'a0': a0, 'zmax': zmax})
        if self.cr_map is None:
            self.cr_map = np.reshape(self.exposure['map'], (1, self.npix))
        else:
            self.cr_map = self.cr_map * self.exposure['map']
        self.cr_map /= np.sum(self.cr_map, axis=-1)[:, np.newaxis]

    def arrival_setup(self, fsig):
        """
        Creates the realizations of the arrival maps.

        :param fsig: signal fraction of cosmic rays per set (signal = originate from sources)
        :type fsig: float
        :return: no return
        """
        dtype = np.uint16 if self.nside <= 64 else np.uint32
        pixel = np.zeros(self.shape).astype(dtype)

        # Setup the signal part
        n_sig = int(fsig * self.ncrs)
        signal_label = np.in1d(range(self.ncrs), np.arange(0, n_sig, 1))
        if n_sig == 0:
            pass
        elif (self.cr_map is None) and (self.sources is None):
            pixel[:, signal_label] = np.random.choice(self.npix, (self.nsets, n_sig))
        elif (self.cr_map is None):
            self._fix_point_source()
        elif np.sum(self.cr_map) > 0:
            if self.cr_map.size == self.npix:
                pixel[:, signal_label] = np.random.choice(self.npix, (self.nsets, n_sig), p=np.hstack(self.cr_map))
            else:
                for i, rig in enumerate(self.rig_bins):
                    mask_rig = (rig == self.rigidities) * signal_label  # type: np.ndarray
                    n = np.sum(mask_rig)
                    if n == 0:
                        continue
                    pixel[mask_rig] = np.random.choice(self.npix, n, p=self.cr_map[i])
        else:
            raise Exception("No signal probability to sample signal from!")

        # Setup the background part
        n_back = self.ncrs - n_sig
        bpdf = self.exposure['map'] if self.exposure['map'] is not None else np.ones(self.npix) / float(self.npix)
        pixel[:, np.invert(signal_label)] = np.random.choice(self.npix, (self.nsets, n_back), p=bpdf)

        self.signal_label = np.repeat(signal_label[np.newaxis], self.nsets, axis=0)
        self.crs['pixel'] = pixel

    def apply_uncertainties(self, err_e=None, err_a=None, err_xmax=None):
        """ Apply measurement uncertainties (see BaseSimulation.apply_uncertainties()). """
        BaseSimulation.apply_uncertainties(self, err_e, err_a, err_xmax)

    def get_data(self, convert_all=False, shuffle=False):
        """
        Returns the data in the form of the cosmic_rays.CosmicRaysSets() container.

        :param convert_all: if True, also vectors and lon/lat of the cosmic ray sets are saved (more memory expensive)
        :type convert_all: bool
        :return: Instance of cosmic_rays.CosmicRaysSets() classes

                 Example:
                 sim = ObservedBound()
                 ...
                 crs = sim.get_data(convert_all=True)
                 pixel = crs['pixel']
                 lon = crs['lon']
                 lat = crs['lat']
                 log10e = crs['log10e']
                 charge = crs['charge']
        """
        if convert_all is True:
            if not hasattr(self.crs, 'lon') or not hasattr(self.crs, 'lat'):
                self.convert_pixel(convert_all=True)
        if shuffle:
            self.shuffle_events()
        self.crs['signal_label'] = self.signal_label
        return self.crs

    def convert_pixel(self, keyword='vecs', convert_all=False):
        """
        Converts pixelized information stored under key 'pixel' to vectors (keyword='vecs')
        or angles (keyword='angles'), accessible via 'lon', 'lat'. When convert_all is True, both are saved.
        This can be used at a later stage, if convert_all was set to False originally.
        """
        shape = (-1, self.shape[0], self.shape[1])
        if self.exposure['map'] is not None:
            a0 = self.exposure['a0']
            zmax = self.exposure['zmax']
            vecs = hpt.rand_exposure_vec_in_pix(self.nside, np.hstack(self.crs['pixel']), a0, zmax).reshape(shape)
        else:
            vecs = hpt.rand_vec_in_pix(self.nside, np.hstack(self.crs['pixel'])).reshape(shape)
        if keyword == 'vecs' or convert_all is True:
            if hasattr(self.crs, 'lon') and hasattr(self.crs, 'lat') and not all:
                raise Exception('Not useful to convert pixels to vecs, information already there!')
            self.crs['vecs'] = vecs
        if keyword == 'angles' or convert_all is True:
            if keyword == 'angles' and not convert_all:
                if hasattr(self.crs, 'vecs') and not convert_all:
                    raise Exception('Not useful to convert pixels to angles, information already there!')
            lon, lat = coord.vec2ang(vecs)
            self.crs['lon'] = lon.reshape(self.shape)
            self.crs['lat'] = lat.reshape(self.shape)
        else:
            raise Exception('keyword not understood! Use angles or vecs!')

    def _fix_point_source(self):
        """ Internal function to set non-smeared sources automatically """
        print("Warning: No extragalactic smearing of the sources performed before lensing (smear_sources). "
              "Sources are considered point-like.")
        eg_map = np.zeros((1, self.npix))
        weights = self.source_fluxes if self.source_fluxes is not None else 1.
        eg_map[:, hpt.vec2pix(self.nside, *self.sources)] = weights
        self.cr_map = eg_map


class SourceBound(BaseSimulation):
    """
    Class to simulate cosmic ray arrival scenario by sources located at the sky, including energies, charges, smearing.
    This is a source bound simulation, thus energies and composition is set at the sources and might differ at Earth.
    """

    def __init__(self, nsets, ncrs):
        """
        Initialization of SourceBound simulation.

        :param nsets: number of simulated cosmic ray sets
        :param ncrs: number of cosmic rays per set
        """
        BaseSimulation.__init__(self, nsets, ncrs)

        self.arrival_matrix, self.source_matrix = None, None
        self.dis_bins, self.log10e_bins = None, None
        self.energy_setting = None
        self.charge_weights = None
        self.universe = SourceGeometry(nsets)

    def set_energy(self, log10e_min, gamma=-2, log10_cut=20., rig_cut=True):
        """
        Define energy spectrum and cut off energy of sources.

        :param log10e_min: Minimum threshold energy for observation
        :param gamma: Spectral index of energy spectrum at sources
        :param log10_cut: Maximum cut-off energy or rigidity for sources
        :param rig_cut: if True, log10_cut refers to a rigidity cut
        """
        self.energy_setting = {'log10e_min': log10e_min, 'gamma': gamma, 'log10_cut': log10_cut, 'rig_cut': rig_cut}

    def set_charges(self, charges):
        """
        Define fraction of charge groups in form of dictionary (e.g. {'h':0.5, 'fe':0.5}) at source
        or as keyword 'first_minimum'/'second_minimum' from Auger's best fit paper (arXiv:1612.07155)
        If string is given, gamma and Rcut are also set to the respective best fit values.

        :param charges: dictionary hosting the fractions of injected elements (H, He, N, Si, Fe possible)
                        or string ('first_minimum'/'second_minimum')
        """
        if isinstance(charges, dict):
            fraction = np.sum([charges[key] for key in charges])
            assert np.abs(fraction - 1) < 1e-4, "Fractions of charges dictionary must be normalized!"
            self.charge_weights = charges

        elif isinstance(charges, str):
            if charges == 'first_minimum':
                self.energy_setting['gamma'], self.energy_setting['log10_cut'] = -0.87, 18.62
                self.charge_weights = {'n': 0.88, 'si': 0.12}
            elif charges == 'second_minimum':
                self.energy_setting['gamma'], self.energy_setting['log10_cut'] = 1.03, 18.21
                self.charge_weights = {'h': 0.6794, 'he': 0.31, 'n': 0.01, 'si': 0.0006}
            elif charges == 'first_minimum_walz':
                self.energy_setting['gamma'], self.energy_setting['log10_cut'] = -0.62, 18.56
                self.charge_weights = {'h': 0.001, 'he': 0.001, 'n': 0.985, 'fe': 0.012}
            elif charges == 'second_minimum_walz':
                self.energy_setting['gamma'], self.energy_setting['log10_cut'] = -2.03, 19.88
                self.charge_weights = {'he': 0.003, 'n': 0.92, 'fe': 0.077}
            else:
                raise Exception('Charge keyword not understood (first_minimum/second_minimum)')
            self.energy_setting['rig_cut'] = True
        else:
            raise Exception('Charge type not understood (dictionary or string)')

    def set_sources(self, source_density, fluxes=None, n_src=100):
        """
        Define source density or directly positions and optional weights (cosmic ray luminosity).

        :param source_density: source density (in 1 / Mpc^3) or array of shape (3, n_sources)
                               that point towards the center of the sources or keyword 'sbg'
        :param fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sources).
        :param n_src: Number of point sources to be considered.
        :return: no return
        """
        self.universe.set_sources(source_density, fluxes, n_src)
        self.crs['sources'] = self.universe.sources

    def attenuate(self, library_path=PATH+'/simulation/crpropa3__emin_18.5__emax_21.npz', inside_fraction=None):
        """
        Apply attenuation for far away sources based on CRPropa simulations

        :param library_path: Input library file to use.
        """
        # Prepare the arrival and source matrix by reweighting
        self._prepare_arrival_matrix(library_path)
        # Assign source allocation of cosmic rays
        self._allocate_sources(inside_fraction)
        # Assign the arrival directions of the cosmic rays
        self._set_arrival_directions()
        # Assign charges and energies of the cosmic rays
        self._set_charges_energies()

    def set_xmax(self, z2a='double', model='EPOS-LHC'):
        """
        Calculate Xmax by Gumbel distribution for the simulated energies and charges.
        For more information refer to BaseSimulation.set_xmax().
        """
        return BaseSimulation.set_xmax(self, z2a, model)

    def smear_sources(self, delta, dynamic=None):
        """
        Smears the source positions with a fisher distribution of width delta (optional dynamic smearing).

        :param delta: either the constant width of the fisher distribution or dynamic (delta / R[10 EV]), in radians
        :param dynamic: if True, function applies dynamic smearing (delta / R[EV]) with value delta at 10 EV rigidity
        :return: no return
        """
        assert self.universe.sources is not None, "Cannot smear sources without positions!"
        assert self.crs is not None, "Cannot smear sources without calling attenuate() before!"

        mask_close = self.crs['source_labels'] >= 0
        d = delta if dynamic is None else delta / \
            10 ** (self.crs['log10e'] - np.log10(self.crs['charge']) - 19.)[mask_close]
        self.crs['vecs'][:, mask_close] = coord.rand_fisher_vec(self.crs['vecs'][:, mask_close], kappa=1/d**2)

    def get_data(self, shuffle=False):
        """
        Returns the data in the form of the cosmic_rays.CosmicRaysSets() container.

        :return: Instance of cosmic_rays.CosmicRaysSets() classes
        """
        if shuffle:
            self.shuffle_events()
        self.crs['signal_label'] = self.signal_label
        return self.crs

    def _prepare_arrival_matrix(self, library_path):
        """
        Prepare the arrival and source matrix by reweighting

        :param library_path: Input library file to use.
        """
        if (self.energy_setting is None) or (self.charge_weights is None):
            raise Exception("You have to define energies and charges before (set_energy() and set_charges())!")

        data = np.load(library_path, allow_pickle=True)
        self.dis_bins, self.log10e_bins = data['distances'], data['log10e_bins']
        if np.median(np.min(self.universe.distances, axis=1)) < np.min(self.dis_bins):
            print("Warning: Distance binning of simulation starts too far outside (%s Mpc)! " % np.min(self.dis_bins))
            print("Required would be: %.2fMpc." % np.median(np.min(self.universe.distances, axis=1)))
        # Assign distance indices for all simulated soures, shape: (self.nsets, self.universe.n_src)
        dis_bin_idx = self.universe.distance_indices(self.dis_bins)

        charge = {'h': 1, 'he': 2, 'n': 7, 'si': 14, 'fe': 26}
        self.source_matrix = np.zeros((self.nsets, self.universe.n_src, 5))
        self.arrival_matrix = np.zeros((self.dis_bins.size, 5, len(self.log10e_bins)-1))
        for key in self.charge_weights:
            f = self.charge_weights[key]
            if f == 0:
                continue
            fractions = data['fractions'].item()[key]
            # reweight to spectral index (simulated gamma=-1) and apply energy / rigidity cut
            fractions = self._reweight_spectrum(fractions, charge[key])
            # as log-space binning the width of the distance bin is increasing with distance
            self.source_matrix += f * np.sum(fractions, axis=-1)[dis_bin_idx]
            self.arrival_matrix += f * fractions
        # Account for optional luminosity weights of sources
        self.source_matrix *= coord.atleast_kd(self.universe.source_fluxes, k=self.source_matrix.ndim)

    def _reweight_spectrum(self, fractions, c):
        """ Internal function to reweight to desired energy spectrum and rigidity cut """
        assert fractions.ndim == 4, "Element arrival matrix fraction must have 4 dimensions!"
        bin_center = (self.log10e_bins[:-1] + self.log10e_bins[1:]) / 2.
        # reweight spectrum (simulated is gamma=-1 as resulting from equally binning in log space)
        fractions *= 10**((self.energy_setting['gamma'] + 1)*(coord.atleast_kd(bin_center, fractions.ndim) - 19))
        # Apply the rigidity cut
        log10_cut = self.energy_setting['log10_cut']
        if self.energy_setting['rig_cut']:
            # convert rigidity cut (log10_cut) to corresponding energy cut (log10e_cut)
            log10e_cut = log10_cut + np.log10(c)
            fcut = np.ones(bin_center.size)
            # cut function from combined fit paper: https://arxiv.org/pdf/1612.07155.pdf
            fcut[bin_center > log10e_cut] = np.exp(1 - 10**(bin_center[bin_center > log10e_cut] - log10e_cut))
            fractions *= coord.atleast_kd(fcut, fractions.ndim)
        else:
            if log10_cut <= self.energy_setting['log10e_min']:
                print("Warning: element with charge Z=%i is not injected with rigidity cut of" % c)
                print("%s and log10e > %s" % (self.energy_setting['log10_cut'], self.energy_setting['log10e_min']))
            fractions[bin_center > log10_cut, :, :, :] = 0
        fractions[:, :, :, bin_center < self.energy_setting['log10e_min']] = 0.  # energy threshold (measured)
        fractions = np.sum(fractions, axis=0)   # sum over injected energies to reweight the spectrum
        return fractions

    def _get_inside_fraction(self):
        """ Internal function to determine fraction of cosmic rays which come from inside rmax (signal cosmic rays) """
        distance_fractions = self.arrival_matrix * coord.atleast_kd(self.dis_bins, 3)
        inside_fraction = np.sum(distance_fractions[self.dis_bins <= self.universe.rmax]) / np.sum(distance_fractions)
        self.crs['inside_fraction'] = inside_fraction
        return inside_fraction

    def _allocate_sources(self, inside_fraction=None):
        """ Internal function to assign the source allocation of the signal (inside rmax) cosmic rays """
        if inside_fraction is None:
            inside_fraction = self._get_inside_fraction()
        # Sample for each set the number of CRs coming from inside and outside rmax
        nsplit = np.random.multinomial(self.ncrs, [inside_fraction, 1-inside_fraction], size=self.nsets).T
        # Assign the CRs from inside rmax to their separate sources (by index label)
        source_labels = -np.ones(self.shape).astype(int)
        n_max = np.max(nsplit[0])
        source_labels[:, :n_max] = sample_from_m_distributions(self.source_matrix.sum(axis=-1), n_max)
        nrange = np.tile(np.arange(self.ncrs), self.nsets).reshape(self.shape)
        mask_close = nrange < nsplit[0][:, np.newaxis]  # Create mask for CRs inside rmax
        source_labels[~mask_close] = -1  # correct the ones resulting by max(nsplit[0])
        self.crs['source_labels'] = source_labels
        occ = np.apply_along_axis(lambda x: np.bincount(x+1)[x+1], axis=1, arr=source_labels)
        self.signal_label = (occ >= 2) & (source_labels >= 0)
        return source_labels

    def _set_arrival_directions(self):
        """ Internal function to sample the arrival directions """
        # First, set random direction of all events
        vecs = coord.rand_vec(self.shape)
        # Set source directions of simulated sources
        mask = self.crs['source_labels'] >= 0   # mask all cosmic rays which originate within rmax
        vecs[:, mask] = self.universe.sources[:, np.argwhere(mask)[:, 0], self.crs['source_labels'][mask]]
        self.crs['vecs'] = vecs
        distances = np.zeros(self.shape)
        distances[mask] = self.universe.distances[np.where(mask)[0], self.crs['source_labels'][mask]]
        self.crs['distances'] = distances
        return vecs

    def _set_charges_energies(self):
        """ Internal function to assign charges and energies of all cosmic rays """
        log10e = np.zeros(self.shape)
        charge = np.zeros(self.shape)
        c = [1, 2, 7, 14, 26]
        d_dis, d_log10e = np.diff(np.log10(self.dis_bins))[0], np.diff(self.log10e_bins)[0]

        # Assign charges and energies of background cosmic rays
        arrival_mat_far = self.arrival_matrix * coord.atleast_kd(self.dis_bins, 3)
        mask_out = self.dis_bins >= self.universe.rmax
        arrival_mat_far[~mask_out] = 0
        arrival_mat_far /= arrival_mat_far.sum()
        mask_close = self.crs['source_labels'] >= 0
        arrival_idx = np.random.choice(arrival_mat_far.size, size=np.sum(~mask_close), p=arrival_mat_far.flatten())
        idx = np.unravel_index(arrival_idx, arrival_mat_far.shape)
        _d = 10**(np.log10(self.dis_bins)[idx[0]] + (np.random.random(idx[0].size) - 0.5) * d_dis)
        _c, _lge = np.array(c)[idx[1]], self.log10e_bins[:-1][idx[2]]
        perm = np.arange(np.sum(~mask_close))
        np.random.shuffle(perm)
        log10e[~mask_close] = _lge[perm]
        charge[~mask_close] = _c[perm]
        self.crs['distances'][~mask_close] = _d[perm]

        # Assign charges and energies of close-by cosmic rays
        dis_bin_idx = self.universe.distance_indices(self.dis_bins)
        for i in range(1 + np.max(dis_bin_idx)):
            # assign distance indices to CRs
            cr_idx = dis_bin_idx[np.where(mask_close)[0], self.crs['source_labels'][mask_close]]
            mask = cr_idx == i
            if np.sum(mask) == 0:
                continue
            w_mat = self.arrival_matrix[i] / self.arrival_matrix[i].sum()
            arrival_idx = np.random.choice(w_mat.size, size=np.sum(mask), p=w_mat.flatten())
            idx = np.unravel_index(arrival_idx, w_mat.shape)
            perm = np.arange(np.sum(mask))
            np.random.shuffle(perm)
            _mask = np.copy(mask_close)
            _mask[_mask] = mask
            charge[_mask], log10e[_mask] = np.array(c)[idx[0]][perm], self.log10e_bins[:-1][idx[1]][perm]
        log10e += d_log10e * np.random.random(self.shape)
        self.crs['log10e'] = log10e
        self.crs['charge'] = charge
        return log10e, charge

    def _get_charge_id(self):
        """ Return charge id of universe """
        chargegroups = ['h', 'he', 'n', 'si', 'fe']
        return ''.join(['__%s_%s' % (key, self.charge_weights[key]) for key in chargegroups
                        if key in self.charge_weights])

    def _select_representative_set(self):  # pragma: no cover
        """ Select a representative set in terms of anisotropies """
        from scipy.stats import mode
        src_labels = np.copy(self.crs['source_labels']).astype(float)
        src_labels[src_labels < 0] = np.nan
        _, counts = mode(src_labels, axis=1, nan_policy='omit')
        return np.argsort(np.squeeze(counts))[int(self.nsets/2.)]

    def _get_strongest_sources(self, idx=None, min_number_src=5):
        """ Select strongest sources """
        ns = np.array([np.sum(self.crs['source_labels'][idx] == k) for k in range(self.universe.n_src)])
        n_t = self.ncrs
        n_thres = ns >= n_t
        while (np.sum(n_thres) < min_number_src) & (n_t >= 3):
            n_t = int(0.8 * n_t)
            n_thres = ns >= n_t
        try:
            src_idx = np.squeeze(np.argwhere(n_thres))[np.argsort(ns[n_thres])]
        except IndexError:
            src_idx = []
        return src_idx, ns

    def plot_spectrum(self, opath=None):  # pragma: no cover
        """ Plot energy spectrum """
        import matplotlib.pyplot as plt
        from scipy.interpolate import interp1d
        from astrotools import statistics as st
        log10e = np.array(self.crs['log10e'])

        plt.figure(figsize=(6, 4))
        dspectrum = auger.SPECTRA_DICT[17]
        log10e_center = dspectrum['logE']
        # log10e_center = log10e_center   # [log10e_center + 0.05 >= self.energy_setting['log10e_min']]
        flux = (10 ** log10e_center) ** 3 * dspectrum['mean']
        flux_high = (10 ** log10e_center) ** 3 * dspectrum['stathi']
        flux_low = (10 ** log10e_center) ** 3 * dspectrum['statlo']
        plt.errorbar(log10e_center, flux, yerr=[flux_low, flux_high], color='red',
                     fmt='.', linewidth=1, markersize=8, capsize=0, label='Auger 2017')

        log10e_bins = np.arange(np.round(np.min(log10e), 1), np.max(log10e) + 0.1, 0.1)
        n, bins = np.histogram(log10e.flatten(), bins=log10e_bins)
        nall = np.apply_along_axis(lambda x: np.histogram(x, bins=log10e_bins)[0], 1, log10e)
        flux_sim = (10 ** st.mid(bins)) ** 2 * n / self.nsets
        flux_unc = (10 ** st.mid(bins)) ** 2 * nall
        norm = flux[np.argmin(np.abs(log10e_center - st.mid(bins)[0]))] / flux_sim[0]
        plt.errorbar(st.mid(bins), norm * flux_sim, marker='.', ls='none', color='k', xerr=0.05,
                     yerr=np.std(norm*flux_unc, axis=0), zorder=-1)
        # plot arriving composition (spline approximation)
        colors = ['firebrick', 'darkorange', 'forestgreen', 'deepskyblue', 'darkblue']
        e, c = ['h', 'he', 'n', 'si', 'fe'], [1, 2, 7, 14, 26]
        for i, ei in enumerate(e):
            mask = self.crs['charge'] == c[i]
            if np.sum(mask) == 0:
                continue
            _n, _bins = np.histogram(log10e[mask].flatten(), bins=log10e_bins)
            _flux_sim = (10 ** st.mid(_bins)) ** 2 * _n / self.nsets
            smooth_spline = interp1d(st.mid(_bins), norm * _flux_sim, kind='cubic', bounds_error=False)
            x = np.linspace(log10e_bins[0], log10e_bins[-1], 100)
            plt.plot(x, smooth_spline(x), color=colors[i], label=ei)
        plt.yscale('log')
        plt.xlim([self.energy_setting['log10e_min'] - 0.01, np.max(log10e) + 0.07])
        plt.ylim([1e36, 1e38])
        plt.legend(loc='lower left', fontsize=12)
        yl = r'$E^{3} \, J(E)$ [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{2}$]'
        plt.ylabel(yl, fontsize=16)
        plt.xlabel(r'$\log_{10}$($E$/eV)', fontsize=16)
        if opath is None:
            opath = '/tmp/spectrum%s__emin_%s__ecut_%s.pdf' % (self._get_charge_id(), self.energy_setting['log10e_min'],
                                                               self.energy_setting['log10_cut'])
        plt.savefig(opath, bbox_inches='tight')
        plt.close()

    def plot_arrivals(self, idx=None, opath=None):  # pragma: no cover
        """ Plot arrival map """
        import matplotlib.pyplot as plt
        from astrotools import skymap
        idx = self._select_representative_set() if idx is None else idx
        src_idx, ns = self._get_strongest_sources(idx)
        labels_p = np.copy(self.crs['source_labels'][idx])
        labels_p[~np.in1d(labels_p, src_idx) & (labels_p >= 0)] = 10*self.universe.n_src
        for j, idxj in enumerate(src_idx):
            labels_p[labels_p == idxj] = j
        cmap = plt.get_cmap('jet', len(src_idx))
        cmap.set_over('k')
        cmap.set_under('gray')
        if len(src_idx) > 0:
            skymap.eventmap(self.crs['vecs'][:, idx], c=labels_p, cmap=cmap, cblabel='Source ID',
                            cticks=np.arange(-1, len(src_idx), 1), vmin=-0.5, vmax=len(src_idx)-0.5)
        else:
            skymap.eventmap(self.crs['vecs'][:, idx], c='gray')
        lon_src, lat_src = coord.vec2ang(self.universe.sources[:, idx])
        plt.scatter(-lon_src, lat_src, c='k', marker='*', s=2*ns)
        ns = np.sort(ns)[::-1]
        plt.title('Strongest sources: (%i, %i, %i)' % (ns[0], ns[1], ns[2]), fontsize=15)
        if opath is None:
            opath = '/tmp/arrival%s__emin_%s__ecut_%s.pdf' % (self._get_charge_id(), self.energy_setting['log10e_min'],
                                                              self.energy_setting['log10_cut'])
        plt.savefig(opath, bbox_inches='tight')
        plt.close()

    def plot_composition_skymap(self, idx=None, opath=None):  # pragma: no cover
        """ Plot arrival map """
        import matplotlib.pyplot as plt
        from astrotools import skymap

        idx = self._select_representative_set() if idx is None else idx
        charges = self.crs['charge'][idx, :]
        cmap = plt.get_cmap('jet_r', np.amax(charges))
        src_idx, ns = self._get_strongest_sources(idx)
        labels_p = np.copy(self.crs['source_labels'][idx])
        labels_p[~np.in1d(labels_p, src_idx) & (labels_p >= 0)] = 10*self.universe.n_src
        for j, idxj in enumerate(src_idx):
            labels_p[labels_p == idxj] = j

        skymap.eventmap(self.crs['vecs'][:, idx, labels_p >= 0], c=charges[labels_p >= 0],
                        cmap=cmap, cblabel='Z',
                        cticks=np.array([1, 2, 6, 12, 20, 26]), vmin=0.5, vmax=26.5,
                        s=25, alpha=0.6, marker='v')

        lons, lats = coord.vec2ang(self.crs['vecs'][:, idx, labels_p < 0])
        plt.scatter(-lons, lats, c=charges[labels_p < 0],
                    cmap=cmap, s=15, alpha=0.4, marker='o', vmin=0.5, vmax=26.5,)

        lon_src, lat_src = coord.vec2ang(self.universe.sources[:, idx])
        plt.scatter(-lon_src, lat_src, c='0.5', linewidth=0.5, edgecolor='k', marker='*', s=7*ns)
        ns = np.sort(ns)[::-1]
        plt.title('Strongest sources: (%i, %i, %i)' % (ns[0], ns[1], ns[2]), fontsize=15)
        if opath is None:
            opath = '/tmp/arrival_charge%s__emin_%s__ecut_%s.pdf' % (self._get_charge_id(),
                                                                     self.energy_setting['log10e_min'],
                                                                     self.energy_setting['log10_cut'])
        plt.savefig(opath, bbox_inches='tight')
        plt.close()

    def plot_distance(self, sets='all', opath=None):  # pragma: no cover
        # pylint: disable=too-many-statements
        """ Plot histogram of travel distances (either mean of all sets or one specific) """
        import matplotlib.pyplot as plt

        dists = self.crs['distances']
        amax = np.amax(dists)
        bin_width = 5 if amax <= 300 else int(amax/50)
        bins = np.arange(-bin_width, np.amax(dists), bin_width)
        bin_centers = bins[1:-1] + bin_width/2.  # without first artificial bin

        if isinstance(sets, int):
            hist_p = np.histogram(dists[sets][self.crs['charge'][sets] == 1], bins)[0][1::]
            hist_he = np.histogram(dists[sets][self.crs['charge'][sets] == 2], bins)[0][1::]
            hist_cno = np.histogram(dists[sets][(self.crs['charge'][sets] > 2) &
                                                (self.crs['charge'][sets] <= 11)], bins)[0][1::]
            hist_medium = np.histogram(dists[sets][(self.crs['charge'][sets] >= 12) &
                                                   (self.crs['charge'][sets] < 16)], bins)[0][1::]
            hist_heavy = np.histogram(dists[sets][self.crs['charge'][sets] >= 17], bins)[0][1::]
            yerr_p, yerr_he, yerr_cno, yerr_medium, yerr_heavy = 0, 0, 0, 0, 0
            amax = np.amax(dists[sets])

        elif sets == 'all':
            # median and std histogram over nsets
            hists_p = np.apply_along_axis(lambda a: np.histogram(a, bins)[0],
                                          1, np.where(self.crs['charge'] == 1, dists,
                                                      np.ones_like(dists)*(-1)))
            hists_he = np.apply_along_axis(lambda a: np.histogram(a, bins)[0],
                                           1, np.where(self.crs['charge'] == 2, dists,
                                                       np.ones_like(dists)*(-1)))
            hists_cno = np.apply_along_axis(lambda a: np.histogram(a, bins)[0],
                                            1, np.where((self.crs['charge'] > 2) &
                                                        (self.crs['charge'] <= 11),
                                                        dists, np.ones_like(dists)*(-1)))
            hists_medium = np.apply_along_axis(lambda a: np.histogram(a, bins)[0], 1,
                                               np.where((self.crs['charge'] >= 12) &
                                                        (self.crs['charge'] < 16),
                                                        dists, np.ones_like(dists)*(-1)))
            hists_heavy = np.apply_along_axis(lambda a: np.histogram(a, bins)[0], 1,
                                              np.where(self.crs['charge'] >= 17,
                                                       dists, np.ones_like(dists)*(-1)))
            hist_p = np.median(hists_p, axis=0)[1::]
            hist_he = np.median(hists_he, axis=0)[1::]
            hist_cno = np.median(hists_cno, axis=0)[1::]
            hist_medium = np.median(hists_medium, axis=0)[1::]
            hist_heavy = np.median(hists_heavy, axis=0)[1::]
            yerr_p = np.std(hists_p, axis=0)[1::]
            yerr_he = np.std(hists_he, axis=0)[1::]
            yerr_cno = np.std(hists_cno, axis=0)[1::]
            yerr_medium = np.std(hists_medium, axis=0)[1::]
            yerr_heavy = np.std(hists_heavy, axis=0)[1::]

        else:
            raise Exception("Sets not understood, either give set id number or keyword all!")

        plt.figure(figsize=(12, 9))
        plt.bar(bin_centers, hist_p,
                color='firebrick', label=r'$Z = 1$', width=bin_width*0.8,
                yerr=yerr_p)
        plt.bar(bin_centers, hist_he,
                color='darkorange', label=r'$Z = 2$', width=bin_width*0.8,
                yerr=yerr_he, bottom=hist_p)
        plt.bar(bin_centers, hist_cno,
                color='forestgreen', label=r'$3 \leq Z \leq 11$', width=bin_width*0.8,
                yerr=yerr_cno, bottom=hist_p+hist_he)
        plt.bar(bin_centers, hist_medium,
                color='deepskyblue', label=r'$12 \leq Z \leq 16$', width=bin_width*0.8,
                bottom=hist_p+hist_he+hist_cno, yerr=yerr_medium)
        plt.bar(bin_centers, hist_heavy,
                bottom=hist_p+hist_he+hist_cno+hist_medium,
                color='darkblue', label=r'$Z \geq 17$', width=bin_width*0.8,
                yerr=yerr_heavy)

        plt.axvline(x=self.universe.rmax, color='0.5', linestyle='dashed', label='Source shell')
        plt.ylabel(r'flux [a.u.] / %s Mpc' % bin_width, fontsize=22)
        plt.xlabel('distance / Mpc', fontsize=22)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.legend(fontsize=20)
        plt.xlim(-0.1, amax)
        plt.ylim(bottom=0)
        plt.grid()
        if opath is None:
            opath = '/tmp/distance%s__emin_%s__ecut_%s__set%s.png' % (self._get_charge_id(),
                                                                      self.energy_setting['log10e_min'],
                                                                      self.energy_setting['log10_cut'], sets)
        plt.savefig(opath, bbox_inches='tight')
        plt.close()
        plt.clf()


class SourceGeometry:
    """
    Class to set up a 3d Universe out of isotropically distributed sources.
    """

    def __init__(self, nsets):
        self.nsets = nsets
        self.rmax = None
        self.n_src = None
        self.sources = None
        self.source_fluxes = None
        self.distances = None

    def set_sources(self, source_density, fluxes, n_src):
        """
        Define source density or directly positions and optional weights (cosmic ray luminosity).

        :param source_density: source density (in 1 / Mpc^3) or array of shape (3, n_sources)
                               that point towards the center of the sources or keyword 'sbg'
        :param fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sources).
        :param n_src: Number of point sources to be considered.
        :return: no return
        """
        self.n_src = n_src
        if isinstance(source_density, (int, float, np.int, np.float)):
            # maximum radius for one source per cosmic ray (isotropy condition)
            self.rmax = (3*n_src/4/np.pi/source_density)**(1/3.)
            self.sources = coord.rand_vec((self.nsets, n_src))  # shape (3, nsets, n_src)
            # random radius in volume
            self.distances = self.rmax * (np.random.random((self.nsets, n_src)))**(1/3.)  # shape (nsets, n_src)
            self.source_fluxes = 1 / self.distances**2
        elif isinstance(source_density, np.ndarray):
            source_density = np.reshape(source_density, (3, -1))
            self.n_src = source_density.shape[1]
            self.sources = source_density
            self.distances = np.sqrt(self.sources**2, axis=0)
            if fluxes is not None:
                assert fluxes.shape == len(source_density[0].shape)
                self.source_fluxes = fluxes
        elif isinstance(source_density, str):
            sources, source_fluxes, distances = getattr(SourceScenario(), source_density.lower())()[:3]
            self.n_src = len(source_fluxes)
            self.sources = np.tile(sources, self.nsets).reshape(sources.shape[0], self.nsets, -1)
            self.source_fluxes = np.tile(source_fluxes, self.nsets).reshape(-1, source_fluxes.shape[0])
            self.distances = np.tile(distances, self.nsets).reshape(-1, distances.shape[0])
            self.rmax = 18.  # outside of most important sbgs -> skymap still quite anisotrop
            #  18 Mpc rmax corresponds to inside fraction around 24 percent
        else:
            raise Exception("Source scenario not understood.")

    def distance_indices(self, distance_bins):
        """
        Get indices of given distance bins in the shape of the sources (nsets, n_src).

        :param distance_bins: Distance bins which refer to indices
        :return indices: indices of distance_bins in shape (nsets, n_src)
        """
        diff = np.abs(np.log(self.distances[np.newaxis] / distance_bins[:, np.newaxis, np.newaxis]))
        return np.argmin(diff, axis=0)


class SourceScenario:
    """Predefined source scenarios"""

    def __init__(self):
        pass

    @staticmethod
    def sbg():
        """Star Burst Galaxy Scenario used in GAP note 2017_007"""
        # Position, fluxes, distances, names of starburst galaxies proposed as UHECRs sources
        # by J. Biteau & O. Deligny (2017)
        # Internal Auger publication: GAP note 2017_007

        lon = np.array([97.4, 141.4, 305.3, 314.6, 138.2, 95.7, 208.7, 106, 240.9, 242, 142.8, 104.9, 140.4, 148.3,
                        141.6, 135.7, 157.8, 172.1, 238, 141.9, 36.6, 20.7, 121.6])
        lat = np.array([-88, 40.6, 13.3, 32, 10.6, 11.7, 44.5, 74.3, 64.8, 64.4, 84.2, 68.6, -17.4, 56.3, -47.4, 24.9,
                        48.4, -51.9, -54.6, 55.4, 53, 27.3, 60.2])
        vecs = coord.ang2vec(np.radians(lon), np.radians(lat))

        distance = np.array([2.7, 3.6, 4, 4, 4, 5.9, 6.6, 7.8, 8.1, 8.1, 8.7, 10.3, 11, 11.4, 15, 16.3, 17.4, 17.9,
                             22.3, 46, 80, 105, 183])
        flux = np.array([13.6, 18.6, 16., 6.3, 5.5, 3.4, 1.1, 0.9, 1.3, 1.1, 2.9, 3.6, 1.7, 0.7, 0.9, 2.6, 2.1, 12.1,
                         1.3, 1.6, 0.8, 1., 0.8])
        names = np.array(['NGC 253', 'M82', 'NGC 4945', 'M83', 'IC 342', 'NGC 6946', 'NGC 2903', 'NGC 5055', 'NGC 3628',
                          'NGC 3627', 'NGC 4631', 'M51', 'NGC 891', 'NGC 3556', 'NGC 660', 'NGC 2146', 'NGC 3079',
                          'NGC 1068', 'NGC 1365', 'Arp 299', 'Arp 220', 'NGC 6240', 'Mkn 231'])

        return vecs, flux, distance, names

    @staticmethod
    def gamma_agn():
        """AGN scenario used in GAP note 2017_007"""
        # Position, fluxes, distances, names of gamma_AGNs proposed as UHECRs sources by J. Biteau & O. Deligny (2017)
        # Internal Auger publication: GAP note 2017_007

        lon = np.array([309.6, 283.7, 150.6, 150.2, 235.8, 127.9, 179.8, 280.2, 63.6, 112.9, 131.9, 98, 340.7, 135.8,
                        160, 243.4, 77.1])
        lat = np.array([19.4, 74.5, -13.3, -13.7, 73, 9, 65, -54.6, 38.9, -9.9, 45.6, 17.7, 27.6, -9, 14.6, -20, 33.5])
        vecs = coord.ang2vec(np.radians(lon), np.radians(lat))

        distance = np.array([3.7, 18.5, 76, 83, 95, 96, 136, 140, 148, 195, 199, 209, 213, 218, 232, 245, 247])
        flux = np.array([0.8, 1, 2.2, 1, 0.5, 0.5, 54, 0.5, 20.8, 3.3, 1.9, 6.8, 1.7, 0.9, 0.4, 1.3, 2.3])
        names = np.array(['Cen A Core', 'M 87', 'NGC 1275', 'IC 310', '3C 264', 'TXS 0149+710', 'Mkn 421',
                          'PKS 0229-581', 'Mkn 501', '1ES 2344+514', 'Mkn 180', '1ES 1959+650', 'AP Librae',
                          'TXS 0210+515', 'GB6 J0601+5315', 'PKS 0625-35', 'I Zw 187'])

        return vecs, flux, distance, names


class CompositionModel:
    """Predefined composition models"""

    def __init__(self, shape, log10e=None):
        self.shape = shape
        self.log10e = log10e

    def mixed(self):
        """Simple estimate of the composition above ~20 EeV by M. Erdmann (2017)"""
        z = {'z': [1, 2, 6, 7, 8], 'p': [0.15, 0.45, 0.4 / 3., 0.4 / 3., 0.4 / 3.]}
        charges = np.random.choice(z['z'], self.shape, p=z['p'])

        return charges

    def mixed_clipped(self):
        """mixed from above, but CNO group only Z=6 because of no lenses at low rigidities"""
        z = {'z': [1, 2, 6], 'p': [0.15, 0.45, 0.4]}
        charges = np.random.choice(z['z'], self.shape, p=z['p'])

        return charges

    def equal(self):
        """Assumes a equal distribution in (H, He, N, Fe) groups."""
        z = {'z': [1, 2, 7, 26], 'p': [0.25, 0.25, 0.25, 0.25]}
        charges = np.random.choice(z['z'], self.shape, p=z['p'])

        return charges

    def auger(self, smoothed=True, model='EPOS-LHC'):
        """Simple estimate from AUGER Xmax measurements"""
        log10e = self.log10e
        charges = auger.rand_charge_from_auger(np.hstack(log10e), model=model,
                                               smoothed=smoothed).reshape(self.shape)

        return charges

    def auger_exponential(self):
        """Simple exponential estimate from AUGER Xmax measurements"""
        log10e = self.log10e
        charges = auger.rand_charge_from_exponential(log10e)

        return charges


class EnergySpectrum:
    """Predefined energy spectra"""

    def __init__(self, shape, log10e_min, log10e_max=20.5):
        self.shape = shape
        self.log10e_min = log10e_min
        self.log10e_max = log10e_max

    def power_law(self, gamma=-3):
        """
        Power law spectrum, with spectral index corresponding to non differential spectrum,
        where gamma=-3.7 corresponds to the AUGER fit at intermediate energies.

        :param gamma: non-differential spectral index (E ~ E^(gamma))
        :return: energies in shape self.shape
        """
        emin = 10**(self.log10e_min - 18.)
        emax = 10**(self.log10e_max - 18.)
        u = np.random.random(self.shape)
        if np.abs(1 + gamma) < 1e-3:
            e = np.exp((np.log(emax) - np.log(emin)) * u + np.log(emin))
        else:
            exp = 1. / (1 + gamma)
            e = ((emax**(1+gamma) - emin**(1+gamma)) * u + emin**(1+gamma))**exp
        return 18. + np.log10(e)

    def auger_fit(self):
        """ Energies following the AUGER spectrum above log10e_min 17.5. """
        return auger.rand_energy_from_auger(self.shape, self.log10e_min, self.log10e_max)
