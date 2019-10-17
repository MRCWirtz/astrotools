import numpy as np
from astrotools import auger, coord, cosmic_rays, healpytools as hpt

print("Test: module cosmic_rays.py")

# This module provides a data container for cosmic ray observables and can be used
# to simply visualize, share, save and load data in an efficient way. There are
# two classes, the CosmicRaysBase and the CosmicRaysSets.

# If you just have a single cosmic ray set you want to use the ComicRaysBase. You can
# set arbitrary content in the container. Objects with shape (self.crs) will be
# stored in an internal array called 'shape_array', all other data in a
# dictionary called 'general_object_store'.
nside = 64
npix = hpt.nside2npix(nside)
ncrs = 5000
exposure = hpt.exposure_pdf(nside)
lon, lat = hpt.pix2ang(nside, hpt.rand_pix_from_map(exposure, n=ncrs))
crs = cosmic_rays.CosmicRaysBase(ncrs)  # Initialize cosmic ray container
# you can set arbitrary content in the container. Objects with different shape
# than (ncrs) will be stored in an internal dictionary called 'general_object_store'
crs['lon'], crs['lat'] = lon, lat
crs['date'] = 'today'
crs['log10e'] = auger.rand_energy_from_auger(log10e_min=19, n=ncrs)
crs.set('vecs', coord.ang2vec(lon, lat))    # another possibility to set content
crs.keys()  # will print the keys that are existing

# Save, load and plot cosmic ray base container
opath = 'cr_base_container.npz'
crs.save(opath)
crs_load = cosmic_rays.CosmicRaysBase(opath)
crs_load.plot_heatmap(opath='cr_base_healpy.png')
crs_load.plot_eventmap(opath='cr_base_eventmap.png')

# You can also quickly write all data in an usual ASCII file:

crs.save_readable('cr_base.txt')

# For a big simulation with a lot of sets (simulated skys), you should use the
# CosmicRaysSets(). Inheriting from CosmicRaysBase(), objects with different
# shape than (nsets, ncrs) will be stored in the 'general_object_store' here.

nsets = 100
crs = cosmic_rays.CosmicRaysSets(nsets, ncrs)
# Objects with different shape than (nsets, ncrs) will be stored in an internal
# dictionary called 'general_object_store'
crs['pixel'] = np.random.randint(0, npix, size=(crs.shape))
crs_set0 = crs[0]           # this indexing will return a CosmicRaysBase() object
crs_subset = crs[10:20]     # will return a subset as CosmicRaysSets() object
