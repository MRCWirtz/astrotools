from astrotools import obs, coord
import healpy
from pylab import *
from time import time


N = 1000
x, y, z = coord.ang2vec(coord.randPhi(N), coord.randTheta(N))
w = np.random.rand(N)

nbins = 30

maxangle = np.deg2rad(30)
abins = np.linspace(0, maxangle, nbins+1)
degs = np.rad2deg((abins[1:] + abins[:-1]) / 2)

maxangle_full = np.pi
abins_full = np.linspace(0, np.pi, nbins+1)
degs_full = np.rad2deg((abins_full[1:] + abins_full[:-1]) / 2)


### 2-point auto correlation: Crosscheck and speed comparison
t1 = time()
ac1 = obs.tpac(x, y, z, maxangle, nbins)
t1 = time() - t1
t2 = time()
ac2 = obs.twoPtAuto(x, y, z, abins)
t2 = time() - t2

print '\n2-point auto-correlation of 1000 isotropic events'
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
# without weights
ac1 = obs.tpac(x, y, z, maxangle_full, nbins, normalized=True)
ac2 = obs.twoPtAuto(x, y, z, abins_full, normalized=True)

figure()
plot(degs_full, ac1, 'b+')
plot(degs_full, ac2, 'r')
ylim(0,1)
title('Normalized Auto-correlation')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')

# with weights
ac1 = obs.tpac(x, y, z, maxangle_full, nbins, normalized=True, weights=w)
ac2 = obs.twoPtAuto(x, y, z, abins_full, normalized=True, weights=w)

figure()
plot(degs_full, ac1, 'b+')
plot(degs_full, ac2, 'r')
ylim(0,1)
title('Normalized Auto-correlation (Weights)')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')


### 2-point cross correlation: Crosscheck and speed comparison
t1 = time()
cc1 = obs.tpcc(x, y, z, x, y, z, maxangle, nbins)
t1 = time() - t1

t2 = time()
cc2 = obs.twoPtCross(x, y, z, x, y, z, abins)
t2 = time() - t2

print '\n2-point cross-correlation of 1000 x 1000 isotropic events'
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
# without weights
cc1 = obs.tpcc(x, y, z, x, y, z, maxangle_full, nbins, normalized=True)
cc2 = obs.twoPtCross(x, y, z, x, y, z, abins_full, normalized=True)

figure()
plot(degs_full, cc1, 'b+')
plot(degs_full, cc2, 'r')
ylim(0,1)
title('Normalized Cross-correlation')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')

# with weights
cc1 = obs.tpcc(x, y, z, x, y, z, maxangle_full, nbins, normalized=True, weights1=w, weights2=w)
cc2 = obs.twoPtCross(x, y, z, x, y, z, abins_full, normalized=True, weights1=w, weights2=w)

figure()
plot(degs_full, cc1, 'b+')
plot(degs_full, cc2, 'r')
ylim(0,1)
title('Normalized Cross-correlation (Weights)')
xlabel('Angular scale [deg]')
ylabel('Correlation signal')


### 2-point correlation on Healpix maps
nside = 64
npix = healpy.nside2npix(nside)
pix = healpy.vec2pix(nside, x,y,z)
m = bincount(pix, minlength=npix, ).astype(float)

cc1 = obs.tpcc(x, y, z, x, y, z, maxangle, nbins)
cc2 = obs.twoPtHealpix(m, m, maxangle, nbins)

print '\n2-point cross-correlation healpix version'
print '  absolute difference tpcc - twoPtHealpix =', cc1 - cc2

figure()
plot(degs, cc1, 'b+')
plot(degs, cc2, 'r')
xlabel('Angular scale [deg]')
ylabel('Number of pairs')
show()