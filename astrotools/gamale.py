import numpy as np
from scipy import sparse
from struct import pack, unpack
import healpy
import healpytools
from bisect import bisect_left
import os


def generateLensPart(fname, nside=64):
    """
    Generate a lens part from the given CRPropa3 file.
    """
    f = np.genfromtxt(fname, names=True)
    row = healpy.vec2pix(nside, f['P0x'], f['P0y'], f['P0z']) # earth
    col = healpy.vec2pix(nside, f['Px'],  f['Py'],  f['Pz'] ) # galaxy
    npix = healpy.nside2npix(nside)
    data = np.ones(len(row))
    M = sparse.coo_matrix((data, (row, col)), shape=(npix, npix))
    M = M.tocsc()
    M /= maxRowSum(M) # normalize rows to account for different number of trajectories
    return M

def saveLensPart(Mcsc, fname):
    """
    Save the given lens part in PARSEC format.
    """
    M = Mcsc.tocoo()
    fout = open(fname, 'wb')
    fout.write(pack('i4', M.nnz))
    fout.write(pack('i4', M.shape[0]))
    fout.write(pack('i4', M.shape[1]))
    data = np.zeros((M.nnz,), dtype=np.dtype([('row', 'i4'), ('col','i4'),('data','f8')]))
    data['row'] = M.row
    data['col'] = M.col
    data['data'] = M.data
    data.tofile(fout)
    fout.close()

def loadLensPart(fname):
    """
    Load a lens part from the given PARSEC file.
    """
    fin = open(fname, 'rb')
    nnz = unpack('i', fin.read(4))[0]
    nrows = unpack('i', fin.read(4))[0]
    ncols = unpack('i', fin.read(4))[0]
    data = np.fromfile(fin, dtype=np.dtype([('row','i4'), ('col','i4'), ('data','f8')]))
    fin.close()
    M = sparse.coo_matrix((data['data'],(data['row'], data['col'])), shape=(nrows, ncols))
    return M.tocsc()

def maxColumnSum(M):
    """
    Return the 1-norm (maximum of absolute sums of columns) of the given matrix.
    """
    return (abs(M)).sum(axis=0).max()

def maxRowSum(M):
    """
    Return the infinity-norm (maximum of sums of absolute rows) of the given matrix.
    """
    return (abs(M)).sum(axis=1).max()

def meanDeflection(Mcsc):
    """
    Calculate the mean deflection of the given matrix
    """
    M = Mcsc.tocoo()
    nside = healpy.npix2nside(M.shape[0])
    return sum(M.data * healpytools.angle(nside, M.row, M.col)) / sum(M.data)

class Lens:
    """
    Galactic magnetic field lens class
    The lens maps directions at the galactic border (pointing inwards) to observed directions on Earth (pointing outwards)
    The galactic coordinate system is used. Angles are avoided.

    The matrices (lensParts) are in compressed sparse column format (scipy.sparse.csc).
    Indices are HEALpixel in ring scheme.
    The row number i indexes the observed direction and the column number j the direction at the Galactic edge.
    """
    lensParts = [] # list of matrices in order of ascending energy
    lRmins = [] # lower rigidity bounds per lens (log10(E/Z/[eV]))
    lRmax = 0 # upper rigidity bound of last lens (log10(E/Z/[eV]))
    nside = None # HEALpix nside parameter
    neutralLensPart = None # matrix for neutral particles

    def load(self, cfname):
        """
        Load and configure the lens from a config file
        filename minR maxR ... in order of ascending rigidity
        """
        dirname = os.path.dirname(cfname)
        data = np.genfromtxt(cfname, dtype=[('fname','S1000'),('E0','f'),('E1','f')])
        for fname, lR0, lR1 in data:
            M = loadLensPart(os.path.join(dirname, fname))
            self.checkLensPart(M)
            self.lensParts.append(M)
            self.lRmins.append(lR0)
            self.lRmax = max(self.lRmax, lR1)
        self.neutralLensPart = sparse.identity(healpy.nside2npix(self.nside), format='csc')

    def checkLensPart(self, M):
        """
        Perform sanity checks and set HEALpix nside parameter.
        """
        nrows, ncols = M.get_shape()
        if nrows != ncols:
            raise Exception("Matrix not square %i x %i"%(nrows, ncols))
        nside = healpy.npix2nside(nrows)
        if self.nside == None:
            self.nside = nside
        elif self.nside != int(nside):
            raise Exception("Matrix have different HEALpix schemes")

    def normalize(self):
        """
        Normalize all matrices to the maximum column sum
        """
        m = 0
        for M in self.lensParts:
            m = max(m, maxColumnSum(M))
        for M in self.lensParts:
            M /= m
        self.neutralLensPart /= m

    def getLensPart(self, E, Z=1):
        """
        Return the matrix corresponding to a given energy E [EeV] and charge number Z
        """
        if Z == 0:
            return self.neutralLensPart
        if len(self.lensParts) == 0:
            raise Exception("Lens empty")        
        lR = np.log10(E / Z) + 18
        if (lR < self.lRmins[0]) or (lR > self.lRmax):
            raise ValueError("Rigidity %f/%i EeV not covered"%(E, Z))
        i = bisect_left(self.lRmins, lR) - 1
        return self.lensParts[i]

    def getObservedVector(self, j, E, Z=1):
        """
        Return the 
        """
        M = self.getLensPart(E, Z)
        v = M.getcol(j).todense()
        return v

    def transformPix(self, j, E, Z=1):
        """
        Attempt to transform a pixel (ring scheme), given an energy E [EeV] and charge number Z.
        Returns a pixel (ring scheme) if successful or None if not.
        """
        M = self.getLensPart(E, Z)
        cmp_val = np.random.rand()
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
        j = healpy.vec2pix(self.nside, x, y, z)
        i = self.transformPix(j, E, Z)
        if i == None:
            return None
        v = healpytools.randVecInPix(self.nside, i)
        return v


import auger
import coord

def applyAugerCoverageToLens(L):
    """
    Apply the Auger exposure to all matrices of a lens and renormalize.
    """
    pix = range(healpy.nside2npix(L.nside))
    v = healpy.pix2vec(L.nside, pix)
    v = coord.gal2eq(*v)
    phi, theta = coord.vec2ang(*v)
    exposure = auger.geometricExposure(theta)
    D = sparse.diags(exposure, 0, format='csc')

    for i, M in enumerate(L.lensParts):
        L.lensParts[i] = D.dot(M)
    L.neutralLensPart = D.dot(M)
    L.normalize()


import matplotlib.pyplot as plt

def plotColSum(M):
    colSums = M.sum(axis=0).tolist()[0]
    plt.plot(colSums, c='b', lw=0.5)

def plotRowSum(M):
    rowSums = M.sum(axis=1).tolist()
    plt.plot(rowSums, c='r', lw=0.5)

def plotMatrix(Mcsc, stride=100):
    M = Mcsc.tocoo()
    plt.scatter(M.col[::stride], M.row[::stride], marker='+')
    plt.xlim(0, M.shape[0])
    plt.ylim(0, M.shape[1])
    plt.xlabel('\#column (extragalactic direction)')
    plt.ylabel('\#row (observed direction)')
