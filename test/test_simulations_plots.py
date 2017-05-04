from astrotools import gamale, simulations
import matplotlib.pyplot as plt

# Loading the lens (enter the lenspath before executing)
lens = gamale.Lens()
lens.load('/path/to/lens.cfg')

nside = 64      # resolution of the HEALPix map (default: 64)
stat = 3        # 3 cosmic ray sets are created
ncrs = 1000     # number of cosmic rays per set

#########################################   SCENARIO 1   #########################################
# Creats a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with energy dependent
# extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad). AUGER's exposure is applied
sim = simulations.CosmicRaySimulation(nside, stat, ncrs)
sim.set_energy(19.3)                        # Set minimum energy of 10^(19.3) eV, and AUGER energy spectrum
sim.set_charges(1.)                         # Set charge to Z=1 (proton)
sim.set_sources('sbg')                      # Keyword for starburst galaxies. May also given an integer for number of
                                            # random placed sources or np.ndarray (x, y, z) of source positions.
sim.set_rigidity_bins(lens)                 # setting rigidity bins (either np.ndarray or the lens)
sim.smear_sources(sigma=0.1, dynamic=True)
sim.apply_exposure()                        # Applying AUGER's exposure
sim.arrival_setup(1.)                       # 100% signal
crs = sim.get_data()                        # Getting the data

crs.plot_eventmap()
plt.savefig('smear_dynamic.png', bbox_inches='tight')
plt.clf()

'''
# For plotting all skymaps:
for i in range(stat):
    crs.plot_eventmap(setid=i)
    plt.savefig('smear_dynamic-%i.png' % i, bbox_inches='tight')
    plt.clf()
    
# Access pixel, lon, lat, energy and charge:
pixel = crs['pixel']
lon = crs['lon']
lat = crs['lat']
log10e = crs['log10e']
charge = crs['charge']
'''

#########################################   SCENARIO 2   #########################################
# Creats a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with energy dependent
# extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad) and galactic magnetic field lensing. AUGER's exposure is applied.
sim = simulations.CosmicRaySimulation(nside, stat, ncrs)
sim.set_energy(19.3)
sim.set_charges(1.)
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.1, dynamic=True)
sim.lensing_map(lens)
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap()
plt.savefig('smear_dynamic-lensed.png', bbox_inches='tight')
plt.clf()

#########################################   SCENARIO 3   #########################################
# Creats a 100% signal mixed composition (45% H, 15% He, 40% CNO) cosmic ray scenario (above 10^19.3 eV) from
# starburst galaxies with energy dependent extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad). AUGER's exposure is applied.
sim = simulations.CosmicRaySimulation(nside, stat, ncrs)
sim.set_energy(19.3)
sim.set_charges('AUGER')
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.1, dynamic=True)
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap()
plt.savefig('smear_dynamic-mixed.png', bbox_inches='tight')
plt.clf()

#########################################   SCENARIO 4   #########################################
# Creats a 100% signal mixed composition cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with energy dependent
# extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad) and galactic magnetic field lensing. AUGER's exposure is applied.
sim = simulations.CosmicRaySimulation(nside, stat, ncrs)
sim.set_energy(19.3)
sim.set_charges('AUGER')
sim.set_sources('sbg')
sim.set_rigidity_bins(lens)
sim.smear_sources(sigma=0.1, dynamic=True)
sim.lensing_map(lens)
sim.apply_exposure()
sim.arrival_setup(1.)
crs = sim.get_data()

crs.plot_eventmap()
plt.savefig('smear_dynamic-lensed-mixed.png', bbox_inches='tight')
plt.clf()