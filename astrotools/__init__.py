# noinspection PyPep8Naming
__all__ = [
    "auger",
    "coord",
    "cosmic_rays",
    "gamale",
    "obs",
    "simulations",
    "skymap",
    "stat",
    "nucleitools",
    "healpytools"]

import astrotools.auger
import astrotools.coord
import astrotools.cosmic_rays
import astrotools.gamale
import astrotools.obs
import astrotools.simulations
import astrotools.skymap
import astrotools.stat
import astrotools.nucleitools
import astrotools.healpytools
import pkg_resources  # part of setuptools


try:
    __version__ = pkg_resources.require("astrotools")[0].version
except pkg_resources.DistributionNotFound:
    __version__ = "Package 'astrotools' not installed!"
