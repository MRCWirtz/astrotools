import numpy as np
import os
import matplotlib.pyplot as plt
from astrotools import auger, coord, cosmic_rays, gamale, healpytools as hpt, simulations, skymap

########################################
# Module: auger.py
########################################
print("Test: module auger.py")

# Analytic parametrization of AUGER energy spectrum
log10e = np.arange(18., 20.5, 0.02)
dN = auger.spectrum_analytic(log10e)
E = 10**(log10e - 18)
E3_dN = E**3 * dN    # multiply with E^3 for better visability

# We sample energies which follow the above energy spectrum
n, emin = 1e7, 18.5     # n: number of drawn samples; emin: 10 EeV; lower energy cut
norm = 4.85e16 * n      # norm to account for solid angle area
log10e_sample = auger.rand_energy_from_auger(n=int(n), log10e_min=emin)
log10e_bins = np.arange(18.5, 20.55, 0.05)
n, bins = np.histogram(log10e_sample, bins=log10e_bins)
E3_dN_sampled = 10**((3-1)*(log10e_bins[:-1]-18)) * n   # -1 for correcting logarithmic bin width

plt.plot(log10e, norm*E3_dN, color='red')
plt.plot(log10e_bins[:-1], E3_dN_sampled, marker='s', color='k', ls='None')
plt.yscale('log')
plt.xlabel('log10(E[eV])', fontsize=16)
plt.ylabel('E$^3$ dN', fontsize=16)
plt.savefig('energy_spectrum.png')
plt.clf()

########################################
# Module: coord.py
########################################
print("Test: module coord.py")

# Creates an isotropic arrival map and convert galactic longitudes (lons) and
# galactic latitudes (lats) into cartesian vectors

ncrs = 3000                        # number of cosmic rays
lons = coord.rand_phi(ncrs)        # isotropic in phi (~Uniform(-pi, pi))
lats = coord.rand_theta(ncrs)      # isotropic in theta (Uniform in cos(theta))
vecs = coord.ang2vec(lons, lats)
log10e = auger.rand_energy_from_auger(n=ncrs, log10e_min=emin)
# Plot an example map with sampled energies. If you specify the opath keyword in
# the skymap function, the plot will be automatically saved and closed
skymap.scatter(vecs, c=log10e, opath='isotropic_sky.png')

# Creates an arrival map with a source located at v_src=(1, 0, 0) and apply a
# fisher distribution around it with gaussian spread sigma=10 degree
v_src = np.array([1, 0, 0])
kappa = 1. / np.radians(10.)**2
vecs = coord.rand_fisher_vec(v_src, kappa=kappa, n=ncrs)
# if you dont specify the opath you can use (fig, ax) to plot more stuff
fig, ax = skymap.scatter(vecs, c=log10e)
plt.scatter(0, 0, s=100, c='red', marker='*')    # plot source in the center
plt.savefig('fisher_single_source_10deg.png', bbox_inches='tight')
plt.close()

# We can also use the coord.rand_fisher_vec() function to apply an angular uncertainty
# on simulated arrival directions by feeding a higher dimensional v_src in shape (3, ncrs).
# Each cosmic ray can also have a separate smearing angle, in the following code snippet
# increasing with the longitude.
lats = np.radians(np.array([-60, -30, -15, 0, 15, 30, 60]))
lons = np.radians(np.arange(-180, 180, 30))
lons, lats = np.meshgrid(lons, lats)
# vectors on this defined grid:
vecs = coord.ang2vec(lons.flatten(), lats.flatten())
# chose longitude dependent uncertainty
sigma = 0.01 + np.abs(lons.flatten()) / (4 * np.pi)
vecs_unc = coord.rand_fisher_vec(vecs, kappa=1/sigma**2)
skymap.scatter(vecs_unc, s=100, c=sigma, cblabel=r'$\sigma$ [rad]')
# To have the reference points we will also visualize the grid (Take care about the different longitude convention here)
plt.scatter(-lons.flatten(), lats.flatten(), marker='+', color='k')
plt.savefig('angular_uncertainty.png', bbox_inches='tight')
plt.close()

########################################
# Module: healpytools.py
########################################
print("Test: module healpytools.py")

# The healpytools provides an extension for the healpy framework (https://healpy.readthedocs.io),
# a tool to pixelize the sphere into cells with equal solid angle. There are various
# functionalities on top of healpy, e.g. sample random directions in pixel or create
# distributions on the sphere (dipole, fisher, experiment exposure).

nside = 64          # resolution of the HEALPix map (default: 64)
npix = hpt.nside2npix(nside)
# Create a dipole distribution for a healpy map
lon, lat = np.radians(45), np.radians(60)   # Position of the maximum of the dipole (healpy and astrotools definition)
vec_max = hpt.ang2vec(lat, lon)             # Convert this to a vector
amplitude = 0.5     # amplitude of dipole
dipole = hpt.dipole_pdf(nside, amplitude, vec_max)
skymap.heatmap(dipole, opath='dipole.png')

# Draw random events from this distribution
pixel = hpt.rand_pix_from_map(dipole, n=ncrs)   # returns 3000 random pixel from this map
vecs = hpt.rand_vec_in_pix(nside, pixel)        # Random vectors within the drawn pixel
skymap.scatter(vecs, c=log10e, opath='dipole_events.png')

# Create a healpy map that follows the exposure of an observatory at latitude
# a0 = -35.25 (Pierre Auger Observatory) and maximum zenith angle of 60 degree
exposure = hpt.exposure_pdf(nside, a0=-35.25, zmax=60)
skymap.heatmap(exposure, opath='exposure.png')

########################################
# Module: cosmic_rays.py
########################################
print("Test: module cosmic_rays.py")

# This module provides a data container for cosmic ray observables and can be used
# to simply share, save and load data. There are two classes, the CosmicRaysBase
# and the CosmicRaysSets

# If you just have a single cosmic ray set you want to use the ComicRaysBase:
ncrs = 5000
lon, lat = hpt.pix2ang(nside, hpt.rand_pix_from_map(exposure, n=ncrs))
crs = cosmic_rays.CosmicRaysBase(ncrs)  # Initialize cosmic ray container
# you can set arbitrary content in the container. Objects with different shape
# than (ncrs) will be stored in an internal dictionary called 'general_object_store'
crs['lon'], crs['lat'] = lon, lat
crs['date'] = 'today'
crs.set('vecs', coord.ang2vec(lon, lat))    # another possibility to set content
crs.keys()  # will print the keys that are existing

# Save, load and plot cosmic ray base container
opath = 'cr_base_container.npz'
crs.save(opath)
crs_load = cosmic_rays.CosmicRaysBase(opath)
crs_load.plot_heatmap(opath='cr_base_healpy.png')
crs_load.plot_eventmap(opath='cr_base_eventmap.png')

# For a big simulation with a lot of sets (skymaps), you can use the CosmicRaysSets()
nsets = 100
crs = cosmic_rays.CosmicRaysSets(nsets, ncrs)
# Objects with different shape than (nsets, ncrs) will be stored in an internal
# dictionary called 'general_object_store'
crs['pixel'] = np.random.randint(0, npix, size=(crs.shape))
crs_set0 = crs[0]           # this indexing will return a CosmicRaysBase() object
crs_subset = crs[10:20]     # will return a subset as CosmicRaysSets() object

########################################
# Module: simulations.py
########################################
print("Test: module simulations.py")

# The simulation module is a tool to setup arrival simulations in a few lines of
# code. It is a wrapper for the core functions and is based on the data container
# provided by the cosmic_rays module.

nsets = 1000    # 1000 cosmic ray sets are created

#########################################   SCENARIO 0   #########################################
# Creates an isotropic map with AUGER energy spectrum above 10 EeV and no charges. AUGER's exposure is applied.
# Initialize the simulation with nsets cosmic ray sets and ncrs cosmic rays in each set
sim = simulations.ObservedBound(nside, nsets, ncrs)
sim.set_energy(log10e_min=19.)                 # Set minimum energy of 10^(19.) eV (10 EeV), and AUGER energy spectrum
sim.apply_exposure()                           # Applying AUGER's exposure
sim.arrival_setup(fsig=0.)                     # 0% signal cosmic rays
crs = sim.get_data()           # Getting the data (object of cosmic_rays.CosmicRaysSets())

crs.plot_eventmap(setid=0)                  # First map of cosmic ray sets is plotted.
plt.savefig('isotropy_auger.png', bbox_inches='tight')
plt.close()
del sim, crs
print("\tScenario 0: Done!")


#########################################   SCENARIO 1   #########################################
# Creates a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with constant
# extragalactic smearing sigma=0.25. AUGER's exposure is applied
sim = simulations.ObservedBound(nside, nsets, ncrs)
sim.set_energy(log10e_min=19.3)             # Set minimum energy of 10^(19.3) eV, and AUGER energy spectrum (20 EeV)
sim.set_charges(charge=1.)                  # Set charge to Z=1 (proton)
sim.set_xmax('double')                      # Sample Xmax values from gumble distribution (assume A = 2*Z)
sim.set_sources(sources='sbg')              # Keyword for starburst galaxies. May also given an integer for number of
                                            # random placed sources or np.ndarray (x, y, z) of source positions.
sim.smear_sources(delta=0.1)                # constant smearing for fisher (kappa = 1/sigma^2)
sim.apply_exposure()                        # Applying AUGER's exposure
sim.arrival_setup(fsig=1.)                  # 100% signal cosmic rays
crs = sim.get_data()                        # Getting the data

crs.plot_eventmap(setid=0)                  # First map of cosmic ray sets is plotted.
plt.savefig('sbg_const_fisher.png', bbox_inches='tight')
plt.close()
del sim, crs
print("\tScenario 1: Done!")


#########################################   SCENARIO 2   #########################################
# Creates a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with rigidity dependent
# extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad). AUGER's exposure is applied
sim = simulations.ObservedBound(nside, nsets, ncrs)
sim.set_energy(19.3)
sim.set_charges(1.)
sim.set_sources('sbg')
sim.set_rigidity_bins(np.arange(17., 20.48, 0.02) - 0.01)  # setting rigidity bins (either np.ndarray or the magnetic field lens)
sim.smear_sources(delta=0.2, dynamic=True)  # dynamic=True for rigidity dependent RMS deflection (sigma / R[10EV])
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap(setid=0)
plt.savefig('sbg_dynamic_fisher.png', bbox_inches='tight')
plt.close()
del sim, crs
print("\tScenario 2: Done!")

# If you have a galactic field lens on your computer, you can execute the following code:
lens_path = '/path/to/config/file.cfg'
if os.path.exists(lens_path):
    lens = gamale.Lens(lens_path)
    #########################################   SCENARIO 3   #########################################
    # Creates a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with energy dependent
    # extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad) and galactic magnetic field lensing.
    # AUGER's exposure is applied.
    sim = simulations.ObservedBound(nside, nsets, ncrs)
    sim.set_energy(19.3)
    sim.set_charges(1.)
    sim.set_sources('sbg')
    sim.set_rigidity_bins(lens)
    sim.smear_sources(delta=0.2, dynamic=True)
    sim.lensing_map(lens)                       # Applying galactic magnetic field deflection
    sim.apply_exposure()
    sim.arrival_setup(1.)
    crs = sim.get_data()

    crs.plot_eventmap(setid=0)
    plt.savefig('sbg_dynamic_fisher_lensed.png', bbox_inches='tight')
    plt.close()
    del sim, crs
    print("\tScenario 3: Done!")


    #########################################   SCENARIO 4   #########################################
    # Creates a 100% signal mixed composition (45% H, 15% He, 40% CNO) cosmic ray scenario (above 10^19.3 eV) from
    # starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad).
    # AUGER's exposure is applied.sim = simulations.ObservedBound(nside, nsets, ncrs)
    sim = simulations.ObservedBound(nside, nsets, ncrs)
    sim.set_energy(19.3)
    sim.set_charges('mixed')                    # keyword for the mixed composition
    sim.set_sources('sbg')
    sim.set_rigidity_bins(lens)
    sim.smear_sources(delta=0.2, dynamic=True)
    sim.lensing_map(lens)
    sim.apply_exposure()
    sim.arrival_setup(1.)
    crs = sim.get_data()

    crs.plot_eventmap(setid=0)
    plt.savefig('sbg_dynamic_fisher_lensed_mixed.png', bbox_inches='tight')
    plt.close()
    del sim, crs
    print("\tScenario 4: Done!")


    #########################################   SCENARIO 5   #########################################
    # Creates a 100% signal AUGER composition (Xmax measurements) cosmic ray scenario (above 10^19.3 eV) from
    # starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad).
    sim = simulations.ObservedBound(nside, nsets, ncrs)
    sim.set_energy(19.3)
    sim.set_charges('auger')                    # keyword for the auger composition
    sim.set_sources('sbg')
    sim.set_rigidity_bins(lens)
    sim.smear_sources(delta=0.2, dynamic=True)
    sim.lensing_map(lens)
    sim.apply_exposure()
    sim.arrival_setup(1.)
    crs = sim.get_data()

    crs.plot_eventmap()
    plt.savefig('sbg_dynamic_fisher_lensed_auger.png', bbox_inches='tight')
    plt.close()
    del sim, crs
    print("\tScenario 5: Done!")


    #########################################   SCENARIO 6   #########################################
    # Creates a 100% signal AUGER composition (Xmax measurements) cosmic ray scenario (above 6 EV) from
    # starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad).
    sim = simulations.ObservedBound(nside, nsets, ncrs)
    sim.set_energy(18.78)                       # minimum energy above 6 EeV
    sim.set_charges('auger')                    # keyword for the auger composition
    sim.set_sources('sbg')
    sim.set_rigidity_bins(lens)
    sim.smear_sources(delta=0.2, dynamic=True)
    sim.lensing_map(lens)
    sim.apply_exposure()
    sim.arrival_setup(1.)
    crs = sim.get_data()

    crs.plot_eventmap()
    plt.savefig('sbg_elow_dynamic_fisher_lensed_auger.png', bbox_inches='tight')
    plt.close()

    # Access data from cosmic ray class
    # By default, the simulated crs contain (pixelized) directions stored under the keyword 'pixel'.
    # You can anyways access the vectors or longitudes/latitudes of the respective pixel centers:
    vecs = crs['vecs']
    lons = crs['lon']
    lats = crs['lat']
    print(crs.keys(), 'no keyword named vecs')
    # This is done by internal conversion. If you have exposure or you want to have not only the
    # pixel centers but a correctly randomized exact information, you can either do that directly via
    # crs = sim.get_data(convert_all=True) or later with:
    crs.convert_pixel(keyword='vecs')
    print(crs.keys(), 'now keyword vecs exists!')

    log10e = crs['log10e']
    charge = crs['charge']

    # Plotting skymap as a function of rigidity
    rigidity = log10e - np.log10(charge)
    crs.plot_eventmap(c=rigidity[0], cblabel='Rigidity / log10(R / V)')
    plt.savefig('sbg_elow_dynamic_fisher_lensed_auger_rigs.png', bbox_inches='tight')
    plt.close()
    print("\tScenario 6: Done!")

########################################
# Module: gamale.py
########################################
print("Test: module gamale.py")
# The gamale module is a tool for handling galactic magnetic field lenses. The lenses can be created with
# the lens-factory: https://git.rwth-aachen.de/astro/lens-factory
# Lenses provide information of the deflection of cosmic rays, consisting of matrices mapping an cosmic
# ray's extragalactic origin to the observed direction on earth (matrices of shape Npix x Npix).
# Individual matrices ('lens parts') represent the deflection of particles in a specific rigidity range.
# One lens consists of multiple .npz-files (the lens parts) and a .cfg-file including information about
# the simulation and the rigidity range of the lens parts.

# If you have a galactic field lens on your computer, you can execute the following code:
lens_path = '/path/to/config/file.cfg'
if os.path.exists(lens_path):
    # Loading a lens
    lens = gamale.Lens(lens_path)

    # Loading the lens part corresponding to a particle of energy log10e and charge z
    log10e = 19  # Make sure that the rigidity is covered in your lens
    z = 1
    lens_part = lens.get_lens_part(log10e=log10e, z=z)

    # Alternatively, a lens part can be loaded directly
    lens_part_path = '/path/to/lens/part.npz'
    lens_part = gamale.load_lens_part(lens_part_path)

    nside = gamale.mat2nside(lens_part)  # calculating nside from lens part
    npix = hpt.nside2npix(nside)

    # Compute the observed directions of cosmic rays that arrives from direction of the
    # extragalactic pixel eg_pix after backpropagation from earth. The amount of
    # backpropagated cosmic rays per pixel is found as "Stat" in the .cfg-file.
    eg_pix = np.random.randint(0, npix)
    obs_dist = gamale.observed_vector(lens_part, eg_pix)  # Distribution of shape (Nside,)
    print("A cosmic ray originating from pixel %i is most likely observed in pixel %i." % (eg_pix, np.argmax(obs_dist)))

    # The other direction is also possible. Calculate the distribution of extragalactic
    # directions for cosmic rays arriving in the observed direction 'obs_pix'.
    obs_pix = np.random.randint(0, npix)
    eg_dist = gamale.extragalactic_vector(lens_part, obs_pix)  # Distribution of shape (Nside,)
    print("A cosmic ray observed in pixel %i most likely originated in pixel %i." % (obs_pix, np.argmax(eg_dist)))

    # Calculating the mean deflection
    mean_deflection = gamale.mean_deflection(lens_part)  # in radians
    print("The mean deflection for this rigidity is %.1f degree." % np.rad2deg(mean_deflection))
    # Mean deflection skymap
    deflection_map = gamale.mean_deflection(lens_part, skymap=True)
    skymap.heatmap(np.rad2deg(deflection_map), label='deflection / degree', cmap='jet', opath='deflection_map.png')

    # Using the observed_vector() function, it is possible to calculate the flux / transparancy
    # of the galactic magnetic field outside of the galaxy by computing the sum of all
    # observed rays reaching the earth originating from the extragalactic pixel pix.
    # The larger the amount of flux for that given pixel, the more rays originating from that
    # direction reach the earth

    # brute force calculation of the flux map
    flux = np.zeros(npix)
    for pix in range(npix):
        flux[pix] = np.sum(gamale.observed_vector(lens_part, pix))

    # gamale function to calculate the flux
    flux = gamale.flux_map(lens_part)
    skymap.heatmap(flux, label='Flux [a.u.]', opath='flux_map.png')

    # Finally, an entire probability distributions of extragalactic cosmic rays can be
    # 'lensed' to Earth by a fast matrix multiplication:

    # We create an extragalctic distributions of 30 gaussian source priors
    eg_map = np.zeros(npix)
    for i in range(30):
        v_src = coord.rand_vec()
        sigma = 10 + 10 * np.random.random()
        eg_map += hpt.fisher_pdf(nside, v_src, k=1/np.deg2rad(sigma)**2, sparse=False)
    eg_map /= np.sum(eg_map)  # normalize to probability density distribution
    skymap.heatmap(eg_map, label='p', vmin=0, vmax=np.round(np.max(eg_map), 4), opath='extragalactic_distribution.png')

    # matrix multiplication to obtain an observed map
    obs_map = lens_part.dot(eg_map)
    skymap.heatmap(obs_map, label='p', vmin=0, vmax=np.round(np.max(obs_map), 4), opath='lensed_observed_distribution.png')
