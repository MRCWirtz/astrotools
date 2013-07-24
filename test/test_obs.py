from astrotools import obs, coord
import healpy
from pylab import *
from time import time


N = 1000
x, y, z = coord.ang2vec(coord.randPhi(N), coord.randTheta(N))
w = np.ones(N)


### 2-point auto correlation: Crosscheck and speed comparison
maxangle = np.deg2rad(30)
nbins = 30
abins = np.linspace(0, maxangle, nbins+1)
degs = ( abins[1:] + abins[:-1] ) / 2 * 180 / pi

t1 = time()
ac1 = obs.tpac(x, y, z, maxangle, nbins)
t1 = time() - t1
t2 = time()
ac2 = obs.twoPtAuto(x, y, z, abins)
t2 = time() - t2

print '2-point auto-correlation of 5000 isotropic events'
print '  tpac (C)', t1, 's'
print '  twoPtAuto (numpy)' , t2, 's'
print '  absolute difference', ac1 - ac2

figure()
plot(degs, ac1, 'b+')
plot(degs, ac2, 'r')
title('Auto-correlation')
xlabel('Angular separation [deg]')
ylabel('Number of pairs')


### 2-point auto correlation: Check normalization
maxangle = np.deg2rad(180)
nbins = 30
abins = np.linspace(0, maxangle, nbins+1)
degs = linspace(0, 180, nbins)

# without weights
ac1 = obs.tpac(x, y, z, maxangle, nbins, normalized=True)
ac2 = obs.twoPtAuto(x, y, z, abins, normalized=True)

figure()
plot(degs, ac1, 'b+')
plot(degs, ac2, 'r')
title('Normalized Auto-correlation')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')

# with weights
ac1 = obs.tpac(x, y, z, maxangle, nbins, normalized=True, weights=w)
ac2 = obs.twoPtAuto(x, y, z, abins, normalized=True, weights=w)

figure()
plot(degs, ac1, 'b+')
plot(degs, ac2, 'r')
title('Normalized Auto-correlation (Weights)')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')


### 2-point cross correlation: Crosscheck and speed comparison
maxangle = np.deg2rad(30)
nbins = 30
abins = np.linspace(0, maxangle, nbins+1)
degs = ( abins[1:] + abins[:-1] ) / 2 * 180 / pi

t1 = time()
cc1 = obs.tpcc(x, y, z, x, y, z, maxangle, nbins)
t1 = time() - t1

t2 = time()
cc2 = obs.twoPtCross(x, y, z, x, y, z, abins)
t2 = time() - t2

print '2-point cross-correlation of 5000 x 5000 isotropic events'
print '  tpcc (C)', t1, 's'
print '  twoPtCross (numpy)' , t2, 's'
print '  absolute difference tpcc - twoPtCross =', cc1 - cc2

figure()
plot(degs, cc1, 'b+')
plot(degs, cc2, 'r')
title('Cross-correlation')
xlabel('Angular separation [deg]')
ylabel('Number of pairs')

### 2-point auto correlation: Check normalization
maxangle = np.deg2rad(180)
nbins = 30
abins = np.linspace(0, maxangle, nbins+1)
degs = ( abins[1:] + abins[:-1] ) / 2 * 180 / pi

# without weights
cc1 = obs.tpcc(x, y, z, x, y, z, maxangle, nbins, normalized=True)
cc2 = obs.twoPtCross(x, y, z, x, y, z, abins, normalized=True)

figure()
plot(degs, cc1, 'b+')
plot(degs, cc2, 'r')
title('Normalized Cross-correlation')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')

# with weights
cc1 = obs.tpcc(x, y, z, x, y, z, maxangle, nbins, normalized=True, weights1=w, weights2=w)
cc2 = obs.twoPtCross(x, y, z, x, y, z, abins, normalized=True, weights1=w, weights2=w)

figure()
plot(degs, cc1, 'b+')
plot(degs, cc2, 'r')
title('Normalized Cross-correlation (Weights)')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')



# ### 2-point correlation on Healpix maps: Evaluation
# maxangle = np.deg2rad(30)
# nbins = 30
# abins = np.linspace(0, maxangle, nbins+1)
# degs = ( abins[1:] + abins[:-1] ) / 2 * 180 / pi

# N = 1000
# w = np.ones(N)

# nside = 64
# npix = healpy.nside2npix(nside)

# mcc = np.zeros(nbins)
# mscc = np.zeros(nbins)
# rcc = np.zeros(nbins)

# for i in range(1000):
#     x, y, z = coord.ang2vec(coord.randPhi(N), coord.randTheta(N))

#     cc1 = obs.tpcc(x, y, z, w, x, y, z, w, maxangle, nbins)
#     mcc += cc1
#     mscc += cc1**2

#     pix = healpy.vec2pix(nside, x,y,z)
#     m = bincount(pix, minlength=npix).astype(float)
#     # m = healpy.ud_grade(m, nside*2, power=-2)
#     cc2 = obs.twoPtHealpix(m, m, maxangle, nbins)

#     rcc += abs(cc1 - cc2)

# rcc /= 1000
# mcc /= 1000
# mscc /= 1000
# scc = (mscc - mcc**2)**.5 # standard deviation to compare rcc against

# figure()
# plot(degs, rcc/mcc)
# plot(degs, scc/mcc)
# xlabel('Angular scale [deg]')
# ylabel('Relative deviation')
# show()
