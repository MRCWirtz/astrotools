from numpy import pi, genfromtxt, random, log, log10, dtype, fromfile
from scipy import sparse
import struct
import healpy
import healpytools
from bisect import bisect_left
import os


def loadMatrix(fname):
    """
    Load PARSEC lens matrix and return in sparse column format
    """
    fin = open(fname, 'rb')
    nnz = struct.unpack('i', fin.read(4))[0]
    nrows = struct.unpack('i', fin.read(4))[0]
    ncols = struct.unpack('i', fin.read(4))[0]
    dt = dtype([('i','i4'), ('j','i4'), ('v','f8')])
    data = fromfile(fin, dtype = dt)
    fin.close()
    M = sparse.coo_matrix((data['v'],(data['i'], data['j'])), shape=(nrows, ncols))
    return M.tocsc()

def normalizeMatrix(M):
    M /= M.sum(axis=0).max()

class Lens:
    """
    Galactic magnetic field lens class
    The lens maps directions at the galactic border (pointing inwards) to observed directions on Earth (pointing outwards)
    The galactic coordinate system is used. Angles are avoided.

    The matrices (lensParts) are in sparse column format. Indices are HEALpixel in ring scheme.
    """
    lensParts = [] # list of matrices in order of ascending energy
    lRmins = [] # lower rigidity bounds per lens (log10(E/Z/[eV]))
    lRmax = 0 # upper rigidity bound of last lens (log10(E/Z/[eV]))
    nOrder = None # HEALpix order
    neutralLensPart = None # matrix for neutral particles

    def load(self, cfname):
        """
        Load and configure the lens from a config file
        filename minR maxR ... in order of ascending rigidity
        """
        dirname = os.path.dirname(cfname)
        data = genfromtxt(cfname, dtype=[('fname','S1000'),('E0','f'),('E1','f')])
        for fname, lR0, lR1 in data:
            M = loadMatrix(os.path.join(dirname, fname))
            self.checkLensPart(M)
            self.lensParts.append(M)
            self.lRmins.append(lR0)
            self.lRmax = max(self.lRmax, lR1)
        self.neutralLensPart = sparse.identity(12 * 4**6)

    def checkLensPart(self, M):
        """
        Perform sanity checks and set HEALpix order.
        """
        nrows, ncols = M.get_shape()
        if nrows != ncols:
            raise Exception("Matrix not square %i x %i"%(nrows, ncols))
        nOrder = log(nrows/12) / log(4)
        if not(nOrder.is_integer()):
            raise Exception("Matrix doesn't correspond to any HEALpix scheme")
        if self.nOrder == None:
            self.nOrder = int(nOrder)
        elif self.nOrder != int(nOrder):
            raise Exception("Matrix have different HEALpix schemes")

    def normalize(self):
        """
        Normalize all matrices to the maximum column sum
        """
        m = 0
        for M in self.lensParts:
            m = max(m, M.sum(axis=0).max())
        for M in self.lensParts:
            M /= m

    def getLensPart(self, E, Z=1):
        """
        Return the matrix corresponding to a given energy E [EeV] and charge number Z
        """
        if Z == 0:
            return self.neutralLensPart
        if len(self.lensParts) == 0:
            raise Exception("Lens empty")
        lR = log10(E / Z) + 18
        if (lR < self.lRmins[0]) or (lR > self.lRmax):
            raise ValueError("Energy %f, charge number %i not covered!"%(E, Z))
        i = bisect_left(self.lRmins, lR) - 1
        return self.lensParts[i]

    def transformPix(self, j, E, Z=1):
        """
        Attempt to transform a pixel (ring scheme), given an energy E [EeV] and charge number Z.
        Returns a pixel (ring scheme) if successful or None if not.
        """
        M = self.getLensPart(E, Z)
        cmp_val = random.rand()
        sum_val = 0
        for i in range(M.indptr[j], M.indptr[j+1]):
            sum_val += M.data[i]
            if cmp_val < sum_val:
                return M.indices[i]
        return None

    def transformVec(self, x, y, z, E, Z=1):
        """
        Attempt to transform a galactic direction, given an energy E [EeV] and charge number Z.
        Returns a triple (x,y,z) if successful or None if not.
        """
        j = healpy.vec2pix(2**self.nOrder, x, y, z)
        i = self.transformPix(j, E, Z)
        if i == None:
            return None
        v = healpytools.randVecInPix(self.nOrder, i)
        return v


import auger, coord

def applyAugerCoverageToLense(L):
    """
    Apply the Auger exposure to all matrices of a lens and renormalize.
    """
    npix = 12 * 4**L.nOrder
    nside = 2**L.nOrder

    v = healpy.pix2vec(nside, range(npix))
    v = coord.galactic2Equatorial(*v)
    phi, theta = coord.vec2Ang(*v)
    exposure = map(auger.geometricExposure, theta)
    D = sparse.diags(exposure, 0, format='csc')

    for i, M in enumerate(L.lensParts):
        L.lensParts[i] = D.dot(M)
    L.neutralLensPart = D.dot(M)
    L.normalize()


from matplotlib.pyplot import plot

def plotColSum(M):
    colSums = M.sum(axis=0).tolist()[0]
    plot(colSums, c='b', lw=0.5)

def plotRowSum(M):
    rowSums = M.sum(axis=1).tolist()
    plot(rowSums, c='r', lw=0.5)
