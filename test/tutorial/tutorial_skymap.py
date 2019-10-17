import numpy as np
from astrotools import simulations, skymap

print("Test: module cosmic_rays.py")

# The skymap module have already been presented multiplet times in the tutorial
# so far. Still there is one special visualization of a close-up look, that is
# demonstrated in the following.

# If you have installed the python package 'basemap' you can execute the following code:
try:
    # Assume we have an isotropic skymap
    sim = simulations.ObservedBound(nside=64, nsets=2, ncrs=10000)
    sim.set_energy(log10e_min=19.)                 # Set minimum energy of 10^(19.) eV (10 EeV), and AUGER energy spectrum
    sim.arrival_setup(fsig=0.)                     # 0% signal cosmic rays
    crs = sim.get_data()           # Getting the data (object of cosmic_rays.CosmicRaysSets())

    # Now we can setup a close-up view by specifying the coordinates of the
    # region of interest (roi)
    patch = skymap.PlotSkyPatch(lon_roi=np.deg2rad(30), lat_roi=np.deg2rad(60), r_roi=0.2, title='My SkyPatch')
    mappable = patch.plot_crs(crs, set_idx=0)
    patch.mark_roi()
    patch.plot_grid()
    patch.colorbar(mappable)
    patch.savefig("skypatch.png")
except ImportError:
    pass
