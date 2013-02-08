from numpy import pi, genfromtxt, random, log, dtype, fromfile
from scipy import sparse
import struct
import healpy
import healpytools
import bisect
import os


def loadMatrix(fname):
    fin = open(fname, 'rb')
    nnz = struct.unpack('i', fin.read(4))[0]
    nrows = struct.unpack('i', fin.read(4))[0]
    ncols = struct.unpack('i', fin.read(4))[0]
    dt = dtype([('i','i4'), ('j','i4'), ('v','f8')])
    data = fromfile(fin, dtype = dt)
    fin.close()
    M = sparse.coo_matrix((data['v'],(data['i'], data['j'])), shape=(nrows, ncols))
    return M.tocsc()

class Lens:
    lensParts = []
    minEnergies = []
    maxEnergy = 0
    nOrder = 0

    def load(self, cfname):
        """ Load lens from a config file, assume energy ordering """
        dirname = os.path.dirname(cfname)
        data = genfromtxt(cfname, dtype=[('fname','S1000'),('E0','f'),('E1','f')])
        for fname, E0, E1 in data:
            M = loadMatrix(os.path.join(dirname, fname))
            self.checkLensPart(M)
            self.lensParts.append(M)
            self.minEnergies.append(E0)
            self.maxEnergy = max(self.maxEnergy, E1)

    def checkLensPart(self, M):
        nrows, ncols = M.get_shape()
        if nrows != ncols:
            raise Exception("Matrix not square %i x %i"%(nrows, ncols))
        nOrder = log(nrows/12) / log(4)
        if not(nOrder.is_integer()):
            raise Exception("Matrix doesn't correspond to any HEALpix scheme")
        self.nOrder = int(nOrder)

    def normalize(self):
        """ Normalize all matrices to the maximum column sum """
        m = 0
        for M in self.lensParts:
            m = max(m, M.sum(axis=0).max())
        for M in self.lensParts:
            M /= m

    def getLensPart(self, energy):
        if len(self.lensParts) == 0:
            raise Exception("Lens empty")
        if (energy < self.minEnergies[0]) or (energy > self.maxEnergy):
            raise ValueError("energy %f not covered by lens"%energy)
        i = bisect.bisect_left(self.minEnergies, energy) - 1
        return self.lensParts[i]

    def transformPix(self, energy, j):
        M = self.getLensPart(energy)
        col = M.getcol(j).tocoo()
        cmp_val = random.rand()
        sum_val = 0
        for i, val in zip(col.row, col.data):
            sum_val += val
            if cmp_val < sum_val:
                return i
        return None

    def transformVec(self, energy, x, y, z):
        j = healpy.vec2pix(2**self.nOrder, x, y, z)
        i = self.transformPix(energy, j)
        if i == None:
            return None
        return healpytools.randVecInPix(self.nOrder, i)


import auger, coordinates

def applyAugerExposure(L):
    npix = 12 * 4**L.nOrder
    nside = 2**L.nOrder

    v = healpy.pix2vec(nside, range(npix))
    v = coordinates.galactic2Equatorial(*v)
    r, phi, theta = coordinates.cartesian2Spherical(*v)
    exposure = map(auger.geometricExposure, theta)
    D = sparse.diags(exposure, 0)

    for i,M in enumerate(L.lensParts):
        L.lensParts[i] = D.dot(M)

    L.normalize()


from matplotlib.pyplot import plot

def plotColSum(M):
    colSums = M.sum(axis=0).tolist()[0]
    plot(colSums, c='b', lw=0.5)

def plotRowSum(M):
    rowSums = M.sum(axis=1).tolist()
    plot(rowSums, c='r', lw=0.5)

def normalizeLensPart(M):
    M /= M.sum(axis=0).max()


if __name__ == "__main__":
    print 'foo'
