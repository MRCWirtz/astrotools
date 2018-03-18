# Astrotools

## General description

The astrotools are a lightweight python library for physics of Ultra-high energy
cosmic rays with particular focus on arrival direction tasks. We support
functionalities in the following fields:

* Coordinate transformations
* Cosmic Ray Observables
* Galactic magnetic field lensing
* Setup simulations of arrival scenarios
* Tools for the Pierre Auger Observatory

It is compatible with python2.6+ and python3.

## Interface
All functions work on numpy arrays if possible

__Defaults__
* energy: $`\log_{10}(E / eV)`$
* charges and masses: integers Z, A
* coordinates: Mpc, galactic coordinate system

__Coordinates__:
* Cartesian: x, y, z
* Local spherical:
    * $`\text{azimuth:} \; \; \phi \in [-\pi, \pi) \; \;`$ with 0 pointing in x, $`\pi/2`$ pointing in y direction
    * $`\text{zenith:} \; \; \theta \in [0, \pi] \; \;`$ with 0 pointing in z direction
* Galactic spherical:
    * $`\text{lon:} \; \; \ell \in [-\pi, \pi) \; \;`$ with 0 pointing in x (GC), $`\pi/2`$ pointing in y direction
    * $`\text{lat:} \; \; b \in [\pi/2, -\pi/2] \; \;`$ with $`\pi/2`$ pointing in z direction

## Installation

Clone and install the astrotools by running the following commands:

```bash
git clone git@git.rwth-aachen.de:astro/astrotools.git
cd astrotools
python setup.py install --user
```

The astrotools require additional dependencies ([numpy](https://github.com/numpy/numpy),
[matplotlib](https://github.com/matplotlib/matplotlib), [scipy](https://github.com/scipy/scipy),
[healpy](https://github.com/healpy/healpy)), which you can easily install via pip:

```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install healpy
```
## Getting started
We provide a small tutorial on the astrotools documentation webpage to demonstrate some of the 
basic astrotools functionalties:
http://astro.pages.rwth-aachen.de/astrotools/tutorial.html

## Documentation
For detailed documentation of the respective modules and functions, please have
a look at the official astrotools documentation:
http://astro.pages.rwth-aachen.de/astrotools
