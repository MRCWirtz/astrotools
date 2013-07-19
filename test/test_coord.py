from astrotools import coord
import numpy as np


x_sg = np.array([1, 2, 2, 3])
y_sg = np.array([7, 2, 3, 6])
z_sg = np.array([9, 0, -1, 0])

x_g, y_g, z_g = coord.sgal2gal(x_sg, y_sg, z_sg)
