import numpy as np
import os
from astrotools import coord, gamale, healpytools as hpt, skymap

print("Test: module gamale.py")
# The gamale module is a tool for handling galactic magnetic field lenses. The lenses can be created with
# the lens-factory: https://git.rwth-aachen.de/astro/lens-factory
# Lenses provide information of the deflection of cosmic rays, consisting of matrices mapping an cosmic
# ray's extragalactic origin to the observed direction on Earth (matrices of shape Npix x Npix).
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
