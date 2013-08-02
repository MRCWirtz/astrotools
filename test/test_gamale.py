from scipy import sparse
from astrotools import gamale, coord
import healpy
import matplotlib.pyplot as plt


nside = 16
npix = healpy.nside2npix(nside)


### Test 1: Lens that maps (x,y,z) -> (x,y,z)
M = sparse.identity(npix, format='csc')

L = gamale.Lens()
L.lensParts = [M]
L.lRmins = [17]
L.lRmax = 21
L.nside = nside

success = True

# using transformPix
for i in range(npix):
    j = L.transformPix(i, E=20)
    x1,y1,z1 = healpy.pix2vec(nside, i)
    x2,y2,z2 = healpy.pix2vec(nside, j)
    if (x1*x2) + (y1*y2) + (z1*z2) < 0.999999:
        print 'Fail:', x1, y1, z1, 'and', x2, y2, z2, 'should be identical'
        success = False
        break

# using transformVec
for i in range(npix):
    x1,y1,z1 = healpy.pix2vec(nside, i)
    x2,y2,z2 = L.transformVec(x1,y1,z1, E=20)
    if (x1*x2) + (y1*y2) + (z1*z2) < 0.995:
        print 'Fail:', x1, y1, z1, 'and', x2, y2, z2, 'should be identical'
        success = False
        break

if success:
    print "Test 1: Identity lens --> Success"



### Test 2: Lens that maps (x,y,z) -> (-x,-y,-z)
M = sparse.lil_matrix((npix, npix))
for i in range(npix):
    x,y,z = healpy.pix2vec(nside, i)
    j = healpy.vec2pix(nside, -x, -y, -z)
    M[i,j] = 1

L = gamale.Lens()
L.lensParts = [M.tocsc()]
L.neutralLensPart = sparse.identity(npix, format='csc')
L.lRmins = [17]
L.lRmax = 21
L.nside = nside

success = True

# using transformPix
for i in range(npix):
    j = L.transformPix(i, E=20)
    x1,y1,z1 = healpy.pix2vec(nside, i)
    x2,y2,z2 = healpy.pix2vec(nside, j)
    if (x1*-x2) + (y1*-y2) + (z1*-z2) < 0.999999:
        print 'Fail:', x1, y1, z1, 'and', -x2, -y2, -z2, 'should be identical'
        success = False
        break

# neutral particles should not keep their direction
for i in range(npix):
    j = L.transformPix(i, E=20, Z=0)
    if i != j:
        print 'Fail: Neutral particles should keep their direction'
        success = False
        break

if success:
    print "Test 2: Inverting lens --> Success"



### Test 3: Lens that maps (x,y,z) -> (x,y,z) and includes Auger exposure
M = sparse.diags([1]*npix, 0, format='csc')

L = gamale.Lens()
L.lensParts = [M]
L.lRmins = [17]
L.lRmax = 21
L.nside = nside
gamale.applyAugerCoverageToLens(L)

success = True

nLost = 0
for i in range(npix):
    j = L.transformPix(i, 20)
    if j == None:
        nLost += 1
        continue
    x1,y1,z1 = healpy.pix2vec(nside, i)
    x2,y2,z2 = healpy.pix2vec(nside, j)
    if (x1*x2) + (y1*y2) + (z1*z2) < 0.999999:
        print 'Fail:', x1, y1, z1, 'and', x2, y2, z2, 'should be identical'
        success = False
        break

if success:
    print "Test 3: Identity lens with Auger Exposure --> Success"
    print 'Fraction of lost events', nLost / float(npix)



### Test 4: Lens with random mapping, including Auger exposure
M = sparse.rand(npix, npix, density=0.01, format='csc')

L = gamale.Lens()
L.lensParts = [M]
L.lRmins = [17]
L.lRmax = 21
L.nside = nside

gamale.applyAugerCoverageToLens(L)
L.normalize()

phi, theta = [], []
for i in range(npix):
    v = healpy.pix2vec(nside, i)
    v = L.transformVec(*v, E=20)
    if v == None:
        continue
    p, t = coord.vec2ang(*v)
    phi.append(p)
    theta.append(t)

plt.subplot(111, projection='hammer')
plt.scatter(phi, theta, s=8, lw=0)
plt.show()

print "Test 4: See plot"