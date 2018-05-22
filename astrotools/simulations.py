"""Module helping to setup simulations"""
import numpy as np
from astrotools import auger, coord, cosmic_rays, healpytools as hpt, nucleitools as nt

__author__ = 'Marcus Wirtz'


def set_fisher_smeared_sources(nside, sources, delta, source_fluxes=None):
    """
    Smears the source positions (optional fluxes) with a fisher distribution of width delta.

    :param nside: nside of the HEALPix pixelization (default: 64)
    :type nside: int
    :param sources: array of shape (3, n_sources) that point towards the center of the sources
    :param delta: float or array with same length as sources: width of the fisher distribution (in radians)
    :param source_fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sources).
    :return: healpy map (with npix(nside) entries) for the smeared sources normalized to 1s
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


class ObservedBound:
    """
    Class to simulate cosmic ray arrival scenario by sources located at the sky, including energies, charges, smearing
    and galactic magnetic field effects.
    This is an observed bound simulation, thus energies and composition is set on Earth and might differ at sources.
    """

    def __init__(self, nside, nsets, ncrs):
        """
        Initialization of the object.

        :param nside: nside of the HEALPix pixelization (default: 64)
        :param nsets: number of simulated cosmic ray sets
        :param ncrs: number of cosmic rays per set
        """
        nsets = int(nsets)
        ncrs = int(ncrs)
        self.nside = nside
        self.npix = hpt.nside2npix(nside)
        self.nsets = nsets
        self.ncrs = ncrs
        self.shape = (nsets, ncrs)
        self.crs = cosmic_rays.CosmicRaysSets((nsets, ncrs))
        self.crs['nside'] = nside
        self.sources = None
        self.source_fluxes = None

        self.rigidities = None
        self.rig_bins = None
        self.cr_map = None
        self.lensed = None
        self.exposure = None
        self.signal_idx = None

    def set_energy(self, log10e_min, log10e_max=None):
        """
        Setting the energies of the simulated cosmic ray set.

        :param log10e_min: Either minimum energy (in log10e) for AUGER setup or numpy.array of
                           energies in shape (nsets, ncrs)
        :type log10e_min: Union[np.ndarray, float]
        :param log10e_max: Maximum energy for AUGER setup
        :return: no return
        """
        if isinstance(log10e_min, np.ndarray):
            if log10e_min.shape == self.shape:
                self.crs['log10e'] = log10e_min
            elif log10e_min.size == self.ncrs:
                print("Warning: the same energies have been used for all simulated sets (nsets).")
                self.crs['log10e'] = np.tile(log10e_min, self.nsets).reshape(self.shape)
            else:
                raise Exception("Shape of input energies not in format (nsets, ncrs).")
        elif isinstance(log10e_min, (float, np.float, int, np.int)):
            if (log10e_min < 17.) or (log10e_min > 21.):
                print("Warning: Specified parameter log10e_min below 17 or above 20.5.")
            log10e = auger.rand_energy_from_auger(self.nsets * self.ncrs, log10e_min=log10e_min, log10e_max=log10e_max)
            self.crs['log10e'] = log10e.reshape(self.shape)
        else:
            raise Exception("Input of emin could not be understood.")

    def set_charges(self, charge, **kwargs):
        """
        Setting the charges of the simulated cosmic ray set.

        :param charge: Either charge number that is used or numpy.array of charges in shape (nsets, ncrs) or keyword
        :type: charge: Union[np.ndarray, str, float]
        :return: no return
        """
        if isinstance(charge, np.ndarray):
            if charge.shape == self.shape:
                self.crs['charge'] = charge
            elif charge.size == self.ncrs:
                self.crs['charge'] = np.tile(charge, self.nsets).reshape(self.shape)
            else:
                raise Exception("Shape of input energies not in format (nsets, ncrs).")
        elif isinstance(charge, (float, np.float, int, np.int)):
            self.crs['charge'] = charge
        elif isinstance(charge, str):
            if not hasattr(self.crs, 'log10e'):
                raise Exception("Use function set_energy() before accessing a composition model.")
            self.crs['charge'] = getattr(CompositionModel(self.shape, self.crs['log10e']), charge.lower())(**kwargs)
        else:
            raise Exception("Input of charge could not be understood.")

    def set_xmax(self, z2a='double', model='EPOS-LHC'):
        """
        Calculate Xmax bei gumbel distribution for the simulated energies and charges.

        :param z2a: How the charge is converted to mass number ['double', 'empiric', 'stable', 'abundance']
        :param model: Hadronic interaction for gumbel distribution
        :return: no return
        """
        if (not hasattr(self.crs, 'log10e')) or (not hasattr(self.crs, 'charge')):
            raise Exception("Use function set_energy() and set_charges() before using function set_xmax.")
        mass = getattr(nt.Charge2Mass(self.crs['charge']), z2a)()
        mass = np.hstack(mass) if isinstance(mass, np.ndarray) else mass
        xmax = auger.rand_xmax(np.hstack(self.crs['log10e']), mass, model=model)
        self.crs['xmax'] = np.reshape(xmax, self.shape)

    def set_sources(self, sources, fluxes=None):
        """
        Define source position and optional weights (cosmic ray luminosity).

        :param sources: array of shape (3, n_sources) that point towards the center of the sources or integer for number
                        of random sources or keyword ['sbg']
        :param fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sorces).
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

    def set_rigidity_bins(self, lens_or_bins):
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
            print("Warning: No extragalactic smearing of the sources performed before lensing (smear_sources). Sources "
                  "are considered point-like.")
            eg_map = np.zeros((1, self.npix))
            weights = self.source_fluxes if self.source_fluxes is not None else 1.
            eg_map[:, hpt.vec2pix(self.nside, *self.sources)] = weights
            self.cr_map = eg_map

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
        self.exposure = hpt.exposure_pdf(self.nside, a0, zmax)
        self.cr_map = np.reshape(self.exposure, (1, self.npix)) if self.cr_map is None else self.cr_map * self.exposure
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
        self.signal_idx = np.random.choice(self.ncrs, n_sig, replace=None)
        mask = np.in1d(range(self.ncrs), self.signal_idx)
        if self.cr_map is None:
            pixel[:, mask] = np.random.choice(self.npix, (self.nsets, n_sig))
        else:
            if self.cr_map.size == self.npix:
                pixel[:, mask] = np.random.choice(self.npix, (self.nsets, n_sig), p=np.hstack(self.cr_map))
            else:
                for i, rig in enumerate(self.rig_bins):
                    mask_rig = (rig == self.rigidities) * mask  # type: np.ndarray
                    n = np.sum(mask_rig)
                    if n == 0:
                        continue
                    pixel[mask_rig] = np.random.choice(self.npix, n, p=self.cr_map[i])

        # Setup the background part
        n_back = self.ncrs - n_sig
        bpdf = self.exposure if self.exposure is not None else np.ones(self.npix) / float(self.npix)
        pixel[:, np.invert(mask)] = np.random.choice(self.npix, (self.nsets, n_back), p=bpdf)

        self.crs['pixel'] = pixel

    def get_data(self, convert_all=None):
        """
        Returns the data in the form of the cosmic_rays.CosmicRaysSets() container.

        :param convert_all: if True, also vectors and lon/lat of the cosmic ray sets are saved (more memory expensive)
        :type convert_all: bool
        :return: Instance of cosmic_rays.CosmicRaysSets() classes

                 Example:
                 sim = CosmicRaySimulation()
                 ...
                 crs = sim.get_data()
                 pixel = crs['pixel']
                 lon = crs['lon']
                 lat = crs['lat']
                 log10e = crs['log10e']
                 charge = crs['charge']
        """
        if convert_all is not None:
            vecs = hpt.pix2vec(self.nside, np.hstack(self.crs['pixel']))
            lon, lat = coord.vec2ang(vecs)
            self.crs['x'] = vecs[0].reshape(self.shape)
            self.crs['y'] = vecs[1].reshape(self.shape)
            self.crs['z'] = vecs[2].reshape(self.shape)
            self.crs['lon'] = lon.reshape(self.shape)
            self.crs['lat'] = lat.reshape(self.shape)

        return self.crs


# class GalacticBound:
#     """
#     Class to propagate cosmic ray sets including energies, charges, smearings and galactic magnetic field effects.
#     This is an galactic bound simulation, thus energies and composition is set at sources and differ at Earth.
#     """
#     def __init__(self, nside, crs, bin_log10e=0.1, bin_charge=1):
#         """
#         Initialization of the object.
#
#         :param nside: nside of the HEALPix pixelization (default: 64)
#         :param crs: number of cosmic rays per set
#         """
#         self.nside = nside
#         self.npix = hpt.nside2npix(nside)
#         self.crs = cosmic_rays.CosmicRaysBase(crs)
#         self.ncrs = len(self.crs)
#         self.pixel = self.crs['pixel']
#         self.log10e = self.crs['log10e']
#         self.charge = self.crs['charge']
#
#         self.rigidities = None
#         self.rig_bins = None
#         self.cr_map = None
#         self.lensed = None
#         self.exposure = None


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
    """Predefined compostion models"""
    def __init__(self, shape, log10e):
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
        charges = auger.rand_charge_from_auger(np.hstack(log10e), model=model, smoothed=smoothed).reshape(self.shape)

        return charges
