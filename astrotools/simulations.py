import numpy as np
from astrotools import auger, gamale, healpytools as hpt

__author__ = 'Marcus Wirtz'


def set_fisher_smeared_sources(nside, sources, source_fluxes, sigma):
    '''
    Smears the source positions (optional fluxes) with a fisher distribution of width sigma

    :param nside: nside of the HEALPix pixelization (default: 64)
    :param sources: array of shape (3, n_sources) that point towards the center of the sources
    :param source_fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sorces).
    :param sigma: width of the fisher distribution (in radians)
    :return: healpy map (with npix(nside) entries) for the smeared sources normalized to 1s
    '''
    npix = hpt.nside2npix(nside)
    egMap = np.zeros(npix)
    for i, v_src in enumerate(sources.T):
        pixels, weights = hpt.fisher_pdf(nside, *v_src, k=1./sigma**2)
        if source_fluxes is not None:
            weights *= source_fluxes[i]
        egMap[pixels] += weights
    return egMap / egMap.sum()


class CosmicRaySimulation():

    '''
    Class to simulate cosmic ray sets including energies, charges, smearings and galactic magnetic field effects.
    This is an observed bound simulation, thus energies and composition is set by the user and differ at the sources.
    '''

    def __init__(self, nside, stat, ncrs):
        '''
        :param nside: nside of the HEALPix pixelization (default: 64)
        :param stat: number of simulated cosmic ray sets
        :param ncrs: number of cosmic rays per set
        '''
        self.nside = nside
        self.stat = stat
        self.ncrs = ncrs
        self.shape = (stat, ncrs)
        self.pixel = None
        self.log10e = None
        self.charge = None
        self.sources = None
        self.source_fluxes = None

        self.rigBins = None
        self.egMap = None
        self.arrivalMap = None
        self.exposure = None


    def set_energy(self, emin, emax=None):
        '''
        Setting the energies of the simulated cosmic ray set.
        
        :param emin: Either minimum energy (in log10e) for AUGER setup or numpy.ndarray of energies in shape (stat, ncrs)
        :param emax: Maximum energy for AUGER setup
        :return: no return
        '''
        if isinstance(emin, np.ndarray):
            if emin.shape == self.shape:
                self.log10e = emin
            elif emin.size == self.ncrs:
                print('Warning: the same energies have been used for all simulated sets (stat).')
                self.log10e = np.tile(emin, stat).reshape(self.shape)
            else:
                raise Exception("Shape of input energies not in format (stat, ncrs).")
        elif isinstance(emin, (float, np.float, int, np.int)):
            log10e = auger.rand_energy_from_auger_spectrum(self.stat * self.ncrs, emin=emin, emax=emax).reshape(self.shape)
            self.log10e = log10e
        else:
            raise Exception("Input of emin could not be understood.")


    def set_charges(self, charge):
        '''
        Setting the charges of the simulated cosmic ray set.
        
        :param charge: Either charge number that is used or numpy.ndarray of charges in shape (stat, ncrs) or keyword
        :return: no return
        '''
        if isinstance(charge, np.ndarray):
            if charge.shape == self.shape:
                self.charge = charge
            elif charge.size == self.ncrs:
                print('Warning: the same charges have been used for all simulated sets (stat).')
                self.charge = np.tile(charge, stat).reshape(self.shape)
            else:
                raise Exception("Shape of input energies not in format (stat, ncrs).")
        elif isinstance(charge, (float, np.float, int, np.int)):
            self.charge = charge * np.ones(self.shape)
        elif isinstance(charge, str):
            # TODO: Implement exact AUGER measurements (energy dependent).
            if charge == 'AUGER':
                # Simple estimate of the composition above ~20 EeV by M. Erdmann (2017)
                self.charge = np.random.choice([1, 2, 6, 7, 8], self.shape, p=[0.15, 0.45, 0.4/3., 0.4/3., 0.4/3.])
            else:
                raise Exception("Keyword string for charge could not be understood.")
        else:
            raise Exception("Input of charge could not be understood.")


    def set_sources(self, sources, fluxes=None):

        '''
        Define source position and optional weights (cosmic ray luminosity).
    
        :param sources: array of shape (3, n_sources) that point towards the center of the sources or integer for number
                        of random sources or keyword ['sbg']
        :param fluxes: corresponding cosmic ray fluxes of the sources of shape (n_sorces).
        :return: no return
        '''
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
            if fluxes.size == sources.shape[1]:
                self.source_fluxes = fluxes
            else:
                raise Exception("Fluxes of sources not understood.")


    def set_rigidity_bins(self, lens_or_bins):
        '''
        Defines the bin centers of the rigidities.
        
        :param lens_or_bins: Either the binning of the lens (identically to lens binning) or the lens itself 
        :return: no return
        '''
        if self.rigBins is None:
            if self.log10e is None:
                raise Exception("Cannot define rigidity bins without energies specified.")
            if self.charge is None:
                print("Warning: Assuming energy dependent deflection instead of rigidity dependent (set_charges to avoid)")

            if isinstance(lens_or_bins, np.ndarray):
                bins = lens_or_bins
            else:
                bins_left = np.array(lens_or_bins.lRmins)
                bins = bins_left + (bins_left[1] - bins_left[0]) / 2.
            rigidities = self.log10e - np.log10(self.charge)
            rigs = bins[np.argmin(np.array([np.abs(rigidities - rig) for rig in bins]), axis=0)]
            rigs = rigs.reshape(self.shape)
            self.rigidities = rigs
            self.rigBins = np.unique(rigs)

        return self.rigBins


    def smear_sources(self, sigma, dynamic=False):
        '''
        Smears the source positions (optional fluxes) with a fisher distribution of width sigma (optional dynamic smearing)
        
        :param sigma: either the constant width of the fisher distribution or dynamic (sigma / R[10 EV]), in radians
        :param dynamic: if True, function applies dynamic smearing (sigma / R[EV]) with value sigma at 10 EV rigidity
        :return: no return
        '''
        if self.sources is None:
            raise Exception("Cannot smear sources without positions.")

        if dynamic is not False:
            if self.rigBins is None:
                raise Exception("Cannot dynamically smear sources without rigidity bins (use set_rigidity_bins).")
            egMap = np.zeros((self.rigBins.size, hpt.nside2npix(self.nside)))
            for i, rig in enumerate(self.rigBins):
                sigma_temp = sigma / 10**(rig - 19.)
                egMap[i] = set_fisher_smeared_sources(self.nside, self.sources, self.source_fluxes, sigma_temp)
        else:
            egMap = set_fisher_smeared_sources(self.nside, self.sources, self.source_fluxes, sigma)
        self.egMap = egMap


    def lensing_map(self, lens):
        '''
        Apply a galactic magnetic field to the extragalactic map.
        
        :param lens: Instance of astrotools.gamale.Lens class, for the galactic magnetic field
        :return: no return
        '''

        npix = hpt.nside2npix(self.nside)
        if self.egMap is None:
            print("Warning: There was no extragalactic smearing of the sources performed before lensing (smear_sources).")
            egMap = np.zeros(npix)
            weights = self.source_fluxes if self.source_fluxes is not None else 1.
            egMap[hpt.vec2pix(self.nside, *self.sources)] = weights
            self.egMap = egMap

        if self.rigBins is None:
            self.set_rigidity_bins(lens)

        arrivalMap = np.zeros((self.rigBins.size, npix))
        for i, rig in enumerate(self.rigBins):
            M = lens.get_lens_part(rig)
            egMap_bin = self.egMap if self.egMap.size == npix else self.egMap[i]
            lensed = M.dot(egMap_bin)
            lensed /= np.sum(lensed)
            arrivalMap[i] = lensed

        self.arrivalMap = arrivalMap


    def apply_exposure(self, a0=-35.25, zmax=60):
        '''
        Apply the exposure (coverage) of any experiment (default: AUGER) to the observed maps.

        :param a0: equatorial declination [deg] of the experiment (default: AUGER, a0=-35.25 deg)
        :param zmax: maximum zenith angle [deg] for the events
        :return: no return
        '''
        self.exposure = hpt.exposure_pdf(self.nside, a0, zmax)
        if (self.arrivalMap is None) and (self.egMap is None):
            print("Warning: Neither eg_map nor lensed_map was set before exposure.")
            self.arrivalMap = self.exposure
        elif self.arrivalMap is None:
            print("Warning: lensed_map was not set before exposure.")
            self.arrivalMap = self.egMap * self.exposure
        else:
            self.arrivalMap *= self.exposure

        if self.arrivalMap.size == hpt.nside2npix(self.nside):
            self.arrivalMap /= np.sum(self.arrivalMap)
        else:
            shape = self.arrivalMap.shape
            self.arrivalMap /= np.repeat(np.sum(self.arrivalMap, axis=1), shape[1]).reshape(shape)


    def arrival_setup(self, fsig):
        '''
        Creates the realizations of the arrival maps.

        :param fsig: signal fraction of cosmic rays per set (signal = originate from sources)
        :return: no return
        '''
        npix = hpt.nside2npix(self.nside)
        pixel = np.zeros(self.shape).astype(np.uint16)
        nSig = int(fsig * self.ncrs)
        if (self.arrivalMap is None) and (self.egMap is None):
            print("Warning: Neither eg_map nor lensed_map nor apply_exposure was set before.")
            pixel = np.random.randint(0, npix, self.stat * self.ncrs).reshape(self.shape)
        elif self.arrivalMap is None:
            print('Warning: no arrival map is existing, thus cosmic rays are not deflected in the GMF.')
            if self.egMap.size == npix:
                pixel = np.random.choice(npix, self.shape, p=self.egMap)
            else:
                for i, rig in enumerate(self.rigBins):      # TODO: Repeated code (can be done nicer)
                    mask = rig == self.rigidities
                    n = np.sum(mask)
                    if n == 0:
                        continue
                    pixel[mask] = np.random.choice(npix, n, p=self.egMap[i])
        elif self.arrivalMap.size == npix:
            pixel = np.random.choice(npix, self.shape, p=self.arrivalMap)
        else:
            for i, rig in enumerate(self.rigBins):          # TODO: Here repeated.
                mask = rig == self.rigidities
                n = np.sum(mask)
                if n == 0:
                    continue
                pixel[mask] = np.random.choice(npix, n, p=self.arrivalMap[i])

        nBack = self.ncrs - nSig
        BPDF = self.exposure if self.exposure is not None else np.ones(npix) / float(npix)
        pixel[:, nSig:] = np.random.choice(npix, self.stat * nBack, p=BPDF).reshape((self.stat, nBack))
        self.pixel = pixel
        self.vecs = hpt.pix2vec(self.nside, pixel)


    def get_data(self):
        '''
        Returns the data of the simulation.
        :return: pixel (directions), log10e, charge
        '''

        return self.pixel, self.log10e, self.charge
