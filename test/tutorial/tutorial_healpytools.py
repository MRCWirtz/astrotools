import numpy as np
import os
import matplotlib.pyplot as plt
from astrotools import auger, coord, cosmic_rays, gamale, healpytools as hpt, simulations, skymap

print("Test: module healpytools.py")

# The healpytools provides an extension for the healpy framework (https://healpy.readthedocs.io),
# a tool to pixelize the sphere into cells with equal solid angle. There are various
# functionalities on top of healpy, e.g. sample random directions in pixel or create
# distributions on the sphere (dipole, fisher, experiment exposure).

ncrs = 3000                        # number of cosmic rays
log10e_min = 18.5                  # minimum energy in log10(E / eV)
nside = 64          # resolution of the HEALPix map (default: 64)
npix = hpt.nside2npix(nside)
# Create a dipole distribution for a healpy map
lon, lat = np.radians(45), np.radians(60)   # Position of the maximum of the dipole (healpy and astrotools definition)
vec_max = hpt.ang2vec(lat, lon)             # Convert this to a vector
amplitude = 0.5     # amplitude of dipole
dipole = hpt.dipole_pdf(nside, amplitude, vec_max, pdf=False)
skymap.heatmap(dipole, opath='dipole.png')

# Draw random events from this distribution
pixel = hpt.rand_pix_from_map(dipole, n=ncrs)   # returns 3000 random pixel from this map
vecs = hpt.rand_vec_in_pix(nside, pixel)        # Random vectors within the drawn pixel
skymap.scatter(vecs, c=auger.rand_energy_from_auger(ncrs, log10e_min), opath='dipole_events.png')

# Create a healpy map that follows the exposure of an observatory at latitude
# a0 = -35.25 (Pierre Auger Observatory) and maximum zenith angle of 60 degree
exposure = hpt.exposure_pdf(nside, a0=-35.25, zmax=60)
skymap.heatmap(exposure, opath='exposure.png')

# Note, if you want to sample from the exposure healpy map random vectors you
# have to be careful with the above method hpt.rand_vec_in_pix,
# as the exposure healpy map reads out the exposure value in the pixel centers,
# whereas hpt.rand_vec_in_pix might sample some directions where
# the exposure already dropped to zero. If you want to sample only isoptropic
# arrival directions it is instead recommended to use
# coord.rand_exposure_vec(), or if you can not avoid healpy
# pixelization use <code> hpt.rand_exposure_vec_in_pix </code>.
