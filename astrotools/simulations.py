import numpy as np
from astrotools import auger, coord, cosmic_rays, healpytools as hpt

__author__ = 'Marcus Wirtz'


def set_fisher_smeared_sources(nside, sources, source_fluxes, sigma):
    """
    Smears the source positions (optional fluxes) with a fisher distribution of width sigma.

    :param nside: nside of the HEALPix pixelization (default: 64)
    :type nside: int
    :param sources: array of shape (3, n_sources) that point towards the center of the sources
    :param source_fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sources).
    :param sigma: width of the fisher distribution (in radians)
    :return: healpy map (with npix(nside) entries) for the smeared sources normalized to 1s
    """
    npix = hpt.nside2npix(nside)
    eg_map = np.zeros(npix)
    for i, v_src in enumerate(sources.T):
        pixels, weights = hpt.fisher_pdf(nside, *v_src, k=1./sigma**2)
        if source_fluxes is not None:
            weights *= source_fluxes[i]
        eg_map[pixels] += weights
    return eg_map / eg_map.sum()


class CosmicRaySimulation:
    """
    Class to simulate cosmic ray sets including energies, charges, smearings and galactic magnetic field effects.
    This is an observed bound simulation, thus energies and composition is set by the user and might differ at sources.
    """
    def __init__(self, nside, nsets, ncrs):
        """
        Initialization of the object.
        
        :param nside: nside of the HEALPix pixelization (default: 64)
        :param nsets: number of simulated cosmic ray sets
        :param ncrs: number of cosmic rays per set
        """
        self.nside = nside
        self.nsets = nsets
        self.ncrs = ncrs
        self.shape = (nsets, ncrs)
        self.crs = cosmic_rays.CosmicRaysSets((nsets, ncrs))
        self.sources = None
        self.source_fluxes = None

        self.rigidities = None
        self.rig_bins = None
        self.cr_map = None
        self.lensed = None
        self.exposure = None

    def set_energy(self, log10e_min, log10e_max=None):
        """
        Setting the energies of the simulated cosmic ray set.
        
        :param log10e_min: Either minimum energy (in log10e) for AUGER setup or numpy.array of energies in shape (nsets, ncrs)
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
            log10e = auger.rand_energy_from_auger(self.nsets * self.ncrs, log10e_min=log10e_min, log10e_max=log10e_max)
            self.crs['log10e'] = log10e.reshape(self.shape)
        else:
            raise Exception("Input of emin could not be understood.")

    def set_charges(self, charge):
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
                print('Warning: the same charges have been used for all simulated sets (nsets).')
                self.crs['charge'] = np.tile(charge, self.nsets).reshape(self.shape)
            else:
                raise Exception("Shape of input energies not in format (nsets, ncrs).")
        elif isinstance(charge, (float, np.float, int, np.int)):
            self.crs['charge'] = charge * np.ones(self.shape)
        elif isinstance(charge, str):
            if charge == 'mixed':
                # Simple estimate of the composition above ~20 EeV by M. Erdmann (2017)
                z = {'z': [1, 2, 6, 7, 8], 'p': [0.15, 0.45, 0.4/3., 0.4/3., 0.4/3.]}
                self.crs['charge'] = np.random.choice(z['z'], self.shape, p=z['p'])
            elif (charge == 'AUGER') or (charge == 'auger'):
                if not np.any(self.crs['log10e']):
                    raise Exception("Cannot model energy dependent charges without energies specified.")
                charge = auger.rand_charge_from_auger(np.hstack(self.crs['log10e']), smoothed=True)
                self.crs['charge'] = charge.reshape(self.shape)
            else:
                raise Exception("Keyword string for charge could not be understood (use: 'AUGER').")
        else:
            raise Exception("Input of charge could not be understood.")

    def set_sources(self, sources, fluxes=None):
        """
        Define source position and optional weights (cosmic ray luminosity).
    
        :param sources: array of shape (3, n_sources) that point towards the center of the sources or integer for number
                        of random sources or keyword ['sbg']
        :param fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sorces).
        :return: no return
        """
        if isinstance(sources, np.ndarray):
            self.sources = sources
        elif isinstance(sources, (int, np.int)):
            src_pix = np.random.randint(0, hpt.nside2npix(self.nside), sources)
            self.sources = np.array(hpt.pix2vec(self.nside, src_pix))
        elif isinstance(sources, str):
            if sources == 'sbg':
                # Position and fluxes of starburst galaxies proposed as UHECRs sources by J. Biteau & O. Deligny (2017)
                # GAP note 2017_007
                src_pix = np.array([49131, 8676, 19033, 11616, 19938, 19652, 7461, 864, 2473, 2335, 124, 1657, 31972,
                                    4215, 42629, 14304, 6036, 43945, 44764, 4212, 4920, 13198, 3335])
                src_flux = np.array([13.6, 18.6, 16., 6.3, 5.5, 3.4, 1.1, 0.9, 1.3, 1.1, 2.9, 3.6, 1.7, 0.7, 0.9, 2.6,
                                     2.1, 12.1, 1.3, 1.6, 0.8, 1., 0.8])
                self.sources = np.array(hpt.pix2vec(self.nside, src_pix))
                self.source_fluxes = src_flux
        else:
            raise Exception("Source scenario not understood.")

        if fluxes is not None:
            if np.shape(fluxes) == np.shape(sources):
                self.source_fluxes = fluxes
            else:
                raise Exception("Fluxes of sources not understood.")

    def set_rigidity_bins(self, lens_or_bins):
        """
        Defines the bin centers of the rigidities.
        
        :param lens_or_bins: Either the binning of the lens (left bin borders) or the lens itself 
        :return: no return
        """
        if self.rig_bins is None:
            if not np.any(self.crs['log10e']):
                raise Exception("Cannot define rigidity bins without energies specified.")
            if not np.any(self.crs['charge']):
                print("Warning: Energy dependent deflection instead of rigidity dependent (set_charges to avoid)")

            if isinstance(lens_or_bins, np.ndarray):
                bins = lens_or_bins  # type: np.array
            else:
                bins = np.array(lens_or_bins.lRmins)
            rigidities = self.crs['log10e'] - np.log10(self.crs['charge'])
            idx = np.digitize(rigidities, bins) - 1
            rigs = (bins + (bins[1] - bins[0]) / 2.)[idx]
            rigs = rigs.reshape(self.shape)
            self.rigidities = rigs
            self.rig_bins = np.unique(rigs)

        return self.rig_bins

    def smear_sources(self, sigma, dynamic=False):
        """
        Smears the source positions with a fisher distribution of width sigma (optional dynamic smearing).
        
        :param sigma: either the constant width of the fisher distribution or dynamic (sigma / R[10 EV]), in radians
        :param dynamic: if True, function applies dynamic smearing (sigma / R[EV]) with value sigma at 10 EV rigidity
        :return: no return
        """
        if self.sources is None:
            raise Exception("Cannot smear sources without positions.")

        if dynamic is not False:
            if self.rig_bins is None:
                raise Exception("Cannot dynamically smear sources without rigidity bins (use set_rigidity_bins()).")
            eg_map = np.zeros((self.rig_bins.size, hpt.nside2npix(self.nside)))
            for i, rig in enumerate(self.rig_bins):
                sigma_temp = sigma / 10**(rig - 19.)
                eg_map[i] = set_fisher_smeared_sources(self.nside, self.sources, self.source_fluxes, sigma_temp)
        else:
            eg_map = set_fisher_smeared_sources(self.nside, self.sources, self.source_fluxes, sigma)
        self.cr_map = eg_map

    def lensing_map(self, lens):
        """
        Apply a galactic magnetic field to the extragalactic map.
        
        :param lens: Instance of astrotools.gamale.Lens class, for the galactic magnetic field
        :return: no return
        """
        npix = hpt.nside2npix(self.nside)
        if self.lensed:
            print("Warning: Cosmic Ray maps were already lensed before.")
        if self.cr_map is None:
            print("Warning: No extragalactic smearing of the sources performed before lensing (smear_sources).")
            eg_map = np.zeros(npix)
            weights = self.source_fluxes if self.source_fluxes is not None else 1.
            eg_map[hpt.vec2pix(self.nside, *self.sources)] = weights
            self.cr_map = eg_map

        if self.rig_bins is None:
            self.set_rigidity_bins(lens)

        arrival_map = np.zeros((self.rig_bins.size, npix))
        for i, rig in enumerate(self.rig_bins):
            lp = lens.get_lens_part(rig)
            eg_map_bin = self.cr_map if self.cr_map.size == npix else self.cr_map[i]
            lensed_map = lp.dot(eg_map_bin)
            del lp.data, lp.indptr, lp.indices
            arrival_map[i] = lensed_map / np.sum(lensed_map)

        self.lensed = True
        self.cr_map = arrival_map

    def apply_exposure(self, a0=-35.25, z_max=60):
        """
        Apply the exposure (coverage) of any experiment (default: AUGER) to the observed maps.

        :param a0: equatorial declination [deg] of the experiment (default: AUGER, a0=-35.25 deg)
        :type a0: float, int
        :param z_max: maximum zenith angle [deg] for the events
        :return: no return
        """
        self.exposure = hpt.exposure_pdf(self.nside, a0, z_max)
        if (self.cr_map is None) and (self.lensed is None):
            print("Warning: Neither smearing_sources() nor lensing_map() was set before exposure.")
            self.cr_map = self.exposure
        else:
            self.cr_map *= self.exposure

        if self.cr_map.size == hpt.nside2npix(self.nside):
            self.cr_map /= np.sum(self.cr_map)
        else:
            shape = self.cr_map.shape
            self.cr_map /= np.repeat(np.sum(self.cr_map, axis=1), shape[1]).reshape(shape)

    def arrival_setup(self, fsig, convert_all=None):
        """
        Creates the realizations of the arrival maps.

        :param fsig: signal fraction of cosmic rays per set (signal = originate from sources)
        :type fsig: float
        :param convert_all: if True, also vectors and lon/lat of the cosmic ray sets are saved (more memory expensive)
        :type convert_all: bool
        :return: no return
        """
        npix = hpt.nside2npix(self.nside)
        pixel = np.zeros(self.shape).astype(np.uint16)
        if self.lensed is None:
            print('Warning: no lensing was performed, thus cosmic rays are not deflected in the GMF.')

        # Setup the signal part
        n_sig = int(fsig * self.ncrs)
        signal_idx = np.random.choice(self.ncrs, n_sig, replace=False)
        mask = np.in1d(range(self.ncrs), signal_idx)
        if self.cr_map is None:
            print("Warning: Neither smear_sources(), nor lensing_map(), nor apply_exposure() was called before.")
            pixel[:, mask] = np.random.choice(npix, (self.nsets, n_sig))
        else:
            if self.cr_map.size == npix:
                pixel[:, mask] = np.random.choice(npix, (self.nsets, n_sig), p=self.cr_map)
            else:
                for i, rig in enumerate(self.rig_bins):
                    mask_rig = (rig == self.rigidities) * mask  # type: np.ndarray
                    n = np.sum(mask)
                    if n == 0:
                        continue
                    pixel[mask_rig] = np.random.choice(npix, n, p=self.cr_map[i])

        # Setup the background part
        n_back = self.ncrs - n_sig
        bpdf = self.exposure if self.exposure is not None else np.ones(npix) / float(npix)
        pixel[:, np.invert(mask)] = np.random.choice(npix, (self.nsets, n_back), p=bpdf)

        self.crs['pixel'] = pixel
        if convert_all is not None:
            vecs = hpt.rand_vec_in_pix(self.nside, np.hstack(pixel))
            lon, lat = coord.vec2ang(vecs)
            self.crs['x'] = vecs[0].reshape(self.shape)
            self.crs['y'] = vecs[1].reshape(self.shape)
            self.crs['z'] = vecs[2].reshape(self.shape)
            self.crs['lon'] = lon.reshape(self.shape)
            self.crs['lat'] = lat.reshape(self.shape)

    def get_data(self):
        """
        Returns the data in the form of the cosmic_rays.CosmicRaysSets() container.
        
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
        return self.crs
