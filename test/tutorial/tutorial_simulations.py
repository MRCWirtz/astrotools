import numpy as np
import os
import matplotlib.pyplot as plt
from astrotools import auger, coord, cosmic_rays, gamale, healpytools as hpt, simulations, skymap

print("Test: module simulations.py")

# The simulation module is a tool to setup arrival simulations in a few lines of
# code. It is a wrapper for the core functions and is based on the data container
# provided by the cosmic_rays module.

ncrs = 1000
nside = 64
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
    # AUGER's exposure is applied and a signal fraction of 50 %.
    sim = simulations.ObservedBound(nside, nsets, ncrs)
    sim.set_energy(19.3)
    sim.set_charges(1.)
    sim.set_sources('sbg')
    sim.set_rigidity_bins(lens)
    sim.smear_sources(delta=0.2, dynamic=True)
    sim.lensing_map(lens)                       # Applying galactic magnetic field deflection
    sim.apply_exposure()
    sim.arrival_setup(fsig=0.5)
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
