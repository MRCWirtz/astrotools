from scipy import sparse
from astrotools import gamale, coordinates
import healpy



### Test 1: Lens that maps (x,y,z) -> (x,y,z)
nOrder = 4
nSide = 2**nOrder
nPix = 12 * 4**nOrder
M = sparse.identity(nPix)

L = gamale.Lens()
L.lensParts = [M.tocoo()]
L.minEnergies = [17]
L.maxEnergy = 21
L.nOrder = nOrder

# using transformPix
for i in range(nPix):
    j = L.transformPix(20, i)
    x1,y1,z1 = healpy.pix2vec(nSide, i)
    x2,y2,z2 = healpy.pix2vec(nSide, j)
    if (x1*x2) + (y1*y2) + (z1*z2) < 0.999999:
        print x1, y1, z1, 'and'
        print x2, y2, z2, 'should be identical\n'
        break

# using transformVec
for i in range(nPix):
    x1,y1,z1 = healpy.pix2vec(nSide, i)
    x2,y2,z2 = L.transformVec(20, x1,y1,z1)
    if (x1*x2) + (y1*y2) + (z1*z2) < 0.995:
        print x1, y1, z1, 'and'
        print x2, y2, z2, 'should be identical\n'
        break



### Test 2: Lens that maps (x,y,z) -> (-x,-y,-z)
nOrder = 4
nSide = 2**nOrder
nPix = 12 * 4**nOrder
M = sparse.lil_matrix((nPix, nPix))
for i in range(nPix):
    x,y,z = healpy.pix2vec(nSide, i)
    j = healpy.vec2pix(nSide, -x, -y, -z)
    M[i,j] = 1

L = gamale.Lens()
L.lensParts = [M.tocoo()]
L.minEnergies = [17]
L.maxEnergy = 21
L.nOrder = nOrder

for i in range(nPix):
    j = L.transformPix(20, i)
    x1,y1,z1 = healpy.pix2vec(nSide, i)
    x2,y2,z2 = healpy.pix2vec(nSide, j)
    if (x1*-x2) + (y1*-y2) + (z1*-z2) < 0.999999:
        print x1, y1, z1, 'and'
        print -x2, -y2, -z2, 'should be identical\n'
        break



### Test 3: Lens that maps (x,y,z) -> (x,y,z) and includes Auger exposure
nOrder = 4
nSide = 2**nOrder
nPix = 12 * 4**nOrder
M = sparse.diags([1]*nPix,0)

L = gamale.Lens()
L.lensParts = [M.tocoo()]
L.minEnergies = [17]
L.maxEnergy = 21
L.nOrder = nOrder
gamale.applyAugerExposure(L)

nLost = 0
for i in range(nPix):
    j = L.transformPix(20, i)
    if j == None:
        nLost += 1
        continue
    x1,y1,z1 = healpy.pix2vec(nSide, i)
    x2,y2,z2 = healpy.pix2vec(nSide, j)
    if (x1*x2) + (y1*y2) + (z1*z2) < 0.999999:
        print x1, y1, z1, 'and'
        print x2, y2, z2, 'should be identical\n'
        break

print 'Fraction of lost events', nLost / float(nPix)



### Test 4: Lens with random mapping, including Auger exposure
nOrder = 4
nSide = 2**nOrder
nPix = 12 * 4**nOrder
M = sparse.rand(nPix, nPix, density=0.01, format='coo')

L = gamale.Lens()
L.lensParts = [M]
L.minEnergies = [17]
L.maxEnergy = 21
L.nOrder = nOrder

gamale.applyAugerExposure(L)
L.normalize()

nLost = 0
phi, theta = [], []
for i in range(nPix):
    v = healpy.pix2vec(nSide, i)
    v = L.transformVec(20, *v)
    if v == None:
        nLost += 1
        continue
    r, p, t = coordinates.cartesian2Spherical(*v)
    phi.append(p)
    theta.append(t)

from matplotlib.pyplot import *
subplot(111, projection='hammer')
scatter(phi, theta, s=8, lw=0)
show()
