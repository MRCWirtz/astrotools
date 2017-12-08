# Astrotools

## General description

The astrotools are a lightweight library for astroparticle physics 
using 

* [numpy](https://github.com/numpy/numpy), 
* [matplotlib](https://github.com/matplotlib/matplotlib), 
* [scipy](https://github.com/scipy/scipy) and 
* [healpy](https://github.com/healpy/healpy).

Functionality for the following topics is implemented: 
 
* Coordinate transformations
* Cosmic Ray Observables
* Auger related tools
* Galactic magnetic field lensing
* Setup simulations of arrival scenarios

It is compatible with python2.6+ and python3.

## Interface
All functions work on numpy arrays if possible

__Defaults__
* energy: $`log_{10}(E / eV)`$
* charges and masses: integers Z, A
* coordinates: Mpc, galactic coordinate system

__Coordinates__:
* Cartesian: x, y, z
* Spherical:
    * $`\phi`$ $`(-\pi, \pi)`$ with 0 pointing in x, $`\pi/2`$ pointing in y direction
    * $`\theta`$ $`(\pi/2, -\pi/2)`$ with $`\pi/2`$ pointing in z direction


