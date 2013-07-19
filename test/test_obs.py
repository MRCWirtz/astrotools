from astrotools import obs
import numpy as np


N = 100
x = np.ones(N)
y = np.zeros(N)
z = np.zeros(N)
w = np.ones(N)
maxangle = np.deg2rad(45)
nbins = 45

ac = obs.tpac(x, y, z, w, maxangle, nbins)
cc = obs.tpcc(x, y, z, w, x, y, z, w, maxangle, nbins)