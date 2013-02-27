from astrotools import coord
from numpy import *


x_sg = array([1, 2, 2, 3])
y_sg = array([7, 2, 3, 6])
z_sg = array([9, 0, -1, 0])

x_g, y_g, z_g = coord.supergalactic2Galactic(x_sg, y_sg, z_sg)
