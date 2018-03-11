import numpy as np
import os
import matplotlib.pyplot as plt
from astrotools import auger, coord, gamale, healpytools as hpt, simulations, skymap

########################################
# Module: auger.py
########################################
print("Test: module auger.py")

# Analytic parametrization of AUGER energy spectrum
log10e = np.arange(18., 20., 0.02)
dN = auger.spectrum_analytic(log10e)
E = 10**(log10e - 18)
E3_dN = E**3 * dN    # multiply with E^3 for better visability

# We sample energies which follow the above energy spectrum
n, emin = 1e7, 18.5     # n: number of drawn samples; emin: 10 EeV; lower energy cut
log10e_sample = auger.rand_energy_from_auger(n=int(n), log10e_min=emin)
log10e_bins = np.arange(18.5, 20.55, 0.05)
n, bins = np.histogram(log10e_sample, bins=log10e_bins)
E3_dN_sampled = 10**((3-1)*(log10e_bins[:-1]-18)) * n   # -1 for correcting logarithmic bin width

norm = 4.8e23
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
# Plot an example map with sampled energies
log10e = auger.rand_energy_from_auger(n=ncrs, log10e_min=emin)
skymap.scatter(vecs, log10e, opath='isotropic_sky.png')

# Creates an arrival map with a source located at v_src=(1, 0, 0) and apply a
# fisher distribution around it with gaussian spread sigma=10 degree
v_src = np.array([1, 0, 0])
kappa = 1. / np.radians(10.)**2
vecs = coord.rand_fisher_vec(v_src, kappa=kappa, n=ncrs)
skymap.scatter(vecs, log10e, opath='fisher_single_source_10deg.png')

########################################
# Module: healpytools.py
########################################
print("Test: module healpytools.py")

# Create a dipole distribution for a healpy map

nside = 64          # resolution of the HEALPix map (default: 64)
lon, lat = np.radians(45), np.radians(60)   # Position of the maximum of the dipole (healpy and astrotools definition)
vec_max = hpt.ang2vec(lat, lon)             # Convert this to a vector
amplitude = 0.5     # amplitude of dipole
dipole = hpt.dipole_pdf(nside, amplitude, vec_max)
skymap.skymap(dipole, opath='dipole.png')

# Draw random events from this distribution
pixel = hpt.rand_pix_from_map(dipole, n=ncrs)   # returns 3000 random pixel from this map
vecs = hpt.rand_vec_in_pix(nside, pixel)        # Random vectors within the drawn pixel
skymap.scatter(vecs, log10e, opath='dipole_events.png')

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
sim = simulations.ObservedBound(nside, nsets, ncrs)    # Initialize the simulation with nsets cosmic ray sets and
                                                             # ncrs cosmic rays in each set
sim.set_energy(log10e_min=19.)                 # Set minimum energy of 10^(19.) eV (10 EeV), and AUGER energy spectrum
sim.apply_exposure()                           # Applying AUGER's exposure
sim.arrival_setup(fsig=0.)                     # 0% signal cosmic rays
crs = sim.get_data()                           # Getting the data (object of cosmic_rays.CosmicRaysSets())

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
lens_path = '/path/to/lens.cfg'
if os.path.exists(lens_path):
    lens = gamale.Lens()
    lens.load()
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
    crs = sim.get_data(convert_all=True)

    crs.plot_eventmap()
    plt.savefig('sbg_elow_dynamic_fisher_lensed_auger.png', bbox_inches='tight')
    plt.close()

    # Access data from cosmic ray class
    vecs = zip(crs['x'], crs['y'], crs['z'])
    lons = crs['lon']
    lats = crs['lat']
    log10e = crs['log10e']
    charge = crs['charge']

    # Plotting skymap as a function of rigidity
    rigidity = log10e - np.log10(charge)

    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.9], projection="hammer")
    events = ax.scatter(-lons[0], lats[0], c=rigidity[0], lw=0, s=8, vmin=np.min(rigidity[0]), vmax=np.max(rigidity[0]))

    cbar = plt.colorbar(events, orientation='horizontal', shrink=0.85, pad=0.05, aspect=30)
    cbar.set_label('Rigidity / log10(R / V)', fontsize=26)
    cbar.set_ticks(np.arange(round(np.min(rigidity[0]), 1), round(np.max(rigidity[0]), 1), 0.2))
    cbar.ax.tick_params(labelsize=24)

    plt.xticks(np.arange(-5. / 6. * np.pi, np.pi, np.pi / 6.),
               ['', '', r'90$^{\circ}$', '', '', r'0$^{\circ}$', '', '', r'-90$^{\circ}$', '', ''], fontsize=26)
    # noinspection PyTypeChecker
    plt.yticks([-np.radians(60), -np.radians(30), 0, np.radians(30), np.radians(60)],
               [r'-60$^{\circ}$', r'-30$^{\circ}$', r'0$^{\circ}$', r'30$^{\circ}$', r'60$^{\circ}$'],
               fontsize=26)
    ax.grid(True)
    plt.savefig('sbg_elow_dynamic_fisher_lensed_auger_rigs.png', bbox_inches='tight')
    plt.close()
    print("\tScenario 6: Done!")
