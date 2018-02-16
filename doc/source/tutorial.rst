===================
Astrotools tutorial
===================

This tutorial is meant to give a short overview about the different functionalities
of the astrotools modules.

Module: coord.py
================
This modules converts between different coordinate systems.
The following code snippet will show some basic setup of isotropic arrival
directions in galactic coordinate system.

We create an isotropic arrival map and convert galactic longitudes (lons) and
galactic latitudes (lats) into cartesian vectors.

.. codeblock:: python
    ncrs = 3000                        # number of cosmic rays
    lons = coord.rand_phi(ncrs)        # isotropic in phi (~Uniform(-pi, pi))
    lats = coord.rand_theta(ncrs)      # isotropic in theta (Uniform in cos(theta))
    vecs = coord.ang2vec(lons, lats)
    # Plot an example map with sampled energies
    log10e = auger.rand_energy_from_auger(n=ncrs, log10e_min=emin)
    skymap.scatter(vecs, log10e, opath='isotropic_sky.png')

In the following code we create an arrival map with a source located at
v_src=(1, 0, 0) and apply a fisher distribution around it with gaussian spread
sigma=10 degree

.. codeblock:: python
    v_src = np.array([1, 0, 0])
    kappa = 1. / np.radians(10.)**2
    vecs = coord.rand_fisher_vec(v_src, kappa=kappa, n=ncrs)
    skymap.scatter(vecs, log10e, opath='fisher_single_source_10deg.png')

Module simulations.py
=====================

The simulation module is a tool to setup arrival simulations in a few lines of
code. It is a wrapper for the core functions and is based on the data container
provided by the cosmic_rays module. In the following we show a few examples how
to quickly setup arrival maps.

The simulation module is based on healpy, a tool to pixelize the sphere into
pixels with equally solid angle (https://healpy.readthedocs.io/en/latest/index.html).
Therefore we first have to set the nside resolution parameter of healpy.

.. codeblock:: python
    nside = 64      # resolution of the HEALPix map (default: 64)
    nsets = 1000    # 1000 cosmic ray sets are created

First we will create an isotropic map with AUGER energy spectrum above 10 EeV and no charges.
AUGER's exposure is applied.
.. codeblock:: python
    sim = simulations.ObservedBound(nside, nsets, ncrs)    # Initialize the simulation with nsets cosmic ray sets and
                                                                 # ncrs cosmic rays in each set
    sim.set_energy(log10e_min=19.)                 # Set minimum energy of 10^(19.) eV (10 EeV), and AUGER energy spectrum
    sim.apply_exposure()                           # Applying AUGER's exposure
    sim.arrival_setup(fsig=0.)                     # 0% signal cosmic rays
    crs = sim.get_data()                           # Getting the data (object of cosmic_rays.CosmicRaysSets())

    crs.plot_eventmap(setid=0)                  # First map of cosmic ray sets is plotted.
    plt.show()



Now we create a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with constant
extragalactic smearing sigma=0.25. AUGER's exposure is applied.
.. codeblock:: python
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
    plt.show()


Finally, we create a 100% signal proton cosmic ray scenario (above 10^19.3 eV) from starburst galaxies with rigidity dependent
extragalactic smearing (sigma = 0.1 / (10 * R[EV]) rad). AUGER's exposure is applied
.. codeblock:: python
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
    plt.show()

For usage of the galactic magnetic field lenses please refer to the tutorial.py file.
