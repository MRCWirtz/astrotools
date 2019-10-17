import numpy as np
import matplotlib.pyplot as plt
from astrotools import auger, coord, skymap

print("Test: module coord.py")

# Creates an isotropic arrival map and convert galactic longitudes (lons) and
# galactic latitudes (lats) into cartesian vectors

ncrs = 3000                        # number of cosmic rays
log10e_min = 18.5                  # minimum energy in log10(E / eV)
lons = coord.rand_phi(ncrs)        # isotropic in phi (~Uniform(-pi, pi))
lats = coord.rand_theta(ncrs)      # isotropic in theta (Uniform in cos(theta))
vecs = coord.ang2vec(lons, lats)
log10e = auger.rand_energy_from_auger(n=ncrs, log10e_min=log10e_min)
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
