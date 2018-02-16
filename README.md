# Astrotools

## General description

The astrotools are a lightweight python library for physics of Ultra-high energy
cosmic rays using the dependencies:

* [numpy](https://github.com/numpy/numpy),
* [matplotlib](https://github.com/matplotlib/matplotlib),
* [scipy](https://github.com/scipy/scipy) and
* [healpy](https://github.com/healpy/healpy).

We support functionalities for the following field:

* Coordinate transformations
* Cosmic Ray Observables
* Galactic magnetic field lensing
* Setup simulations of arrival scenarios
* Tools for the Pierre Auger Observatory

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


## Installation

Clone, build and install the astrotools by running the following commands:

```bash
git clone https://git.rwth-aachen.de/astro/astrotools.git
cd astrotools
python setup.py build
python setup.py install --user
```
