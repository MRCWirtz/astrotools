from astrotools import gamale, gamale_sparse, simulations
import matplotlib.pyplot as plt

# Loading the lens (enter your correct lens path before executing)
lens = gamale.Lens()
lens.load('/path/to/lens.cfg')

nside = 64      # resolution of the HEALPix map (default: 64)
nsets = 1000    # 1000 cosmic ray sets are created
ncrs = 3000     # number of cosmic rays per set


#########################################   SCENARIO 0   #########################################
# Creates an isotropic map with AUGER energy spectrum above 10 EeV and no charges. AUGER's exposure is applied.
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)    # Initialize the simulation with nsets cosmic ray sets and
                                                             # ncrs cosmic rays in each set
sim.set_energy(log10e_min=19.)                 # Set minimum energy of 10^(19.) eV (10 EeV), and AUGER energy spectrum
sim.apply_exposure()                           # Applying AUGER's exposure
sim.arrival_setup(fsig=0.)                     # 0% signal cosmic rays
crs = sim.get_data()                           # Getting the data (object of cosmic_rays.CosmicRaysSets())

crs.plot_eventmap(setid=0)                  # First map of cosmic ray sets is plotted.
plt.savefig('isotropy_auger.png', bbox_inches='tight')
plt.close()
del sim, crs
print("Scenario 0: Done!")


#########################################   SCENARIO 1   #########################################
# Creates a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with constant
# extragalactic smearing sigma=0.25. AUGER's exposure is applied
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim.set_energy(log10e_min=19.3)             # Set minimum energy of 10^(19.3) eV, and AUGER energy spectrum (20 EeV)
sim.set_charges(charge=1.)                  # Set charge to Z=1 (proton)
sim.set_sources(sources='sbg')              # Keyword for starburst galaxies. May also given an integer for number of
                                            # random placed sources or np.ndarray (x, y, z) of source positions.
sim.smear_sources(sigma=0.1)                # constant smearing for fisher (kappa = 1/sigma^2)
sim.apply_exposure()                        # Applying AUGER's exposure
sim.arrival_setup(fsig=1.)                  # 100% signal cosmic rays
crs = sim.get_data()                        # Getting the data

crs.plot_eventmap(setid=0)                  # First map of cosmic ray sets is plotted.
plt.savefig('sbg_const_fisher.png', bbox_inches='tight')
plt.close()
del sim, crs
print("Scenario 1: Done!")


#########################################   SCENARIO 2   #########################################
# Creates a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with rigidity dependent
# extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad). AUGER's exposure is applied
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim.set_energy(19.3)
sim.set_charges(1.)
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)                 # setting rigidity bins (either np.ndarray or the lens)
sim.smear_sources(sigma=0.2, dynamic=True)  # dynamic=True for rigidity dependent RMS deflection (sigma / R[10EV])
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap(setid=0)
plt.savefig('sbg_dynamic_fisher.png', bbox_inches='tight')
plt.close()
del sim, crs
print("Scenario 2: Done!")


#########################################   SCENARIO 3   #########################################
# Creates a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with energy dependent
# extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad) and galactic magnetic field lensing.
# AUGER's exposure is applied.
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim.set_energy(19.3)
sim.set_charges(1.)
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.2, dynamic=True)
sim.lensing_map(lens)                       # Applying galactic magnetic field deflection
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap(setid=0)
plt.savefig('sbg_dynamic_fisher_lensed.png', bbox_inches='tight')
plt.close()
del sim, crs
print("Scenario 3: Done!")


#########################################   SCENARIO 4   #########################################
# Creates a 100% signal mixed composition (45% H, 15% He, 40% CNO) cosmic ray scenario (above 10^19.3 eV) from
# starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad).
# AUGER's exposure is applied.sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim.set_energy(19.3)
sim.set_charges('mixed')                    # keyword for the mixed composition
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.2, dynamic=True)
sim.lensing_map(lens)
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap(setid=0)
plt.savefig('sbg_dynamic_fisher_lensed_mixed.png', bbox_inches='tight')
plt.close()
del sim, crs
print("Scenario 4: Done!")


#########################################   SCENARIO 5   #########################################
# Creates a 100% signal AUGER composition (Xmax measurements) cosmic ray scenario (above 10^19.3 eV) from
# starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad).
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim.set_energy(19.3)
sim.set_charges('auger')                    # keyword for the auger composition
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.2, dynamic=True)
sim.lensing_map(lens)
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap()
plt.savefig('sbg_dynamic_fisher_lensed_auger.png', bbox_inches='tight')
plt.close()
del sim, crs
print("Scenario 5: Done!")


#########################################   SCENARIO 6   #########################################
# Creates a 100% signal AUGER composition (Xmax measurements) cosmic ray scenario (above 6 EV) from
# starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad).
sim = simulations.CosmicRaySimulation(nside, nsets, ncrs)
sim.set_energy(18.78)                       # minimum energy above 6 EeV
sim.set_charges('auger')                    # keyword for the auger composition
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.2, dynamic=True)
sim.lensing_map(lens)
sim.apply_exposure()
sim.arrival_setup(1., convert_all=True)
crs = sim.get_data()

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
import numpy as np
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
print("Scenario 6: Done!")
