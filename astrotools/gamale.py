"""
Galactic magnetic field lens
see PARSEC: A Parametrized Simulation Engine for Ultra-High Energy Cosmic Ray Protons, arXiv:1302.3761
http://web.physik.rwth-aachen.de/Auger_MagneticFields/PARSEC/
"""
import gzip
import os
from bisect import bisect_left
from struct import pack, unpack

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import astrotools.healpytools as hpt

# python 2/3 compatibility
try:
    basestring
except NameError:
    basestring = str  # pylint: disable=W0622,C0103


def max_column_sum(mat):
    """
    Return the 1-norm (maximum of absolute sums of columns) of the given matrix.
    The absolute value can be omitted, since the matrix elements are all positive.
    """
    return mat.sum(axis=0).max()


def max_row_sum(mat):
    """
    Return the infinity-norm (maximum of sums of absolute rows) of the given matrix.
    The absolute value can be omitted, since the matrix elements are all positive.
    """
    return mat.sum(axis=1).max()


def normalize_row_sum(mat_csc):
    """
    Normalize each row of a CSC matrix to a row sum of 1.
    """
    row_sum = np.array(mat_csc.sum(axis=1).transpose())[0]
    mat_csc.data /= row_sum[mat_csc.indices]


def generate_lens_part(fname, nside=64):
    """
    Generate a lens part from the given CRPropa3 file.
    """
    # noinspection PyTypeChecker
    f = np.genfromtxt(fname, names=True)
    row = hpt.vec2pix(nside, f['P0x'], f['P0y'], f['P0z'])  # earth
    col = hpt.vec2pix(nside, f['Px'], f['Py'], f['Pz'])  # galaxy
    npix = hpt.nside2npix(nside)
    data = np.ones(len(row))
    mat = sparse.coo_matrix((data, (row, col)), shape=(npix, npix))
    mat = mat.tocsc()
    normalize_row_sum(mat)
    return mat


def save_lens_part(mat_csc, fname):
    """
    Save the lens part in PARSEC format (coordinate type sparse format).
    """
    mat = mat_csc.tocoo()
    fout = open(fname, 'wb')
    fout.write(pack('i4', mat.nnz))
    fout.write(pack('i4', mat.shape[0]))
    fout.write(pack('i4', mat.shape[1]))
    data = np.zeros((mat.nnz,), dtype=np.dtype([('row', 'i4'), ('col', 'i4'), ('data', 'f8')]))
    data['row'] = mat.row
    data['col'] = mat.col
    data['data'] = mat.data
    data.tofile(fout)
    fout.close()


def load_lens_part(fname):
    """
    Load a lens part from the given PARSEC file.
    """
    zipped = fname.endswith(".gz")
    if zipped:
        fin = gzip.open(fname, 'rb')
    else:
        fin = open(fname, 'rb')

    _ = unpack('i', fin.read(4))[0]         # Do not delete this line! (Pops first 4 bytes)
    nrows = unpack('i', fin.read(4))[0]
    ncols = unpack('i', fin.read(4))[0]
    if zipped:
        data = np.frombuffer(fin.read(), dtype=np.dtype([('row', 'i4'), ('col', 'i4'), ('data', 'f8')]))
    else:
        data = np.fromfile(fin, dtype=np.dtype([('row', 'i4'), ('col', 'i4'), ('data', 'f8')]))
    fin.close()
    mat = sparse.coo_matrix((data['data'], (data['row'], data['col'])), shape=(nrows, ncols))
    return mat.tocsc()


def mean_deflection(mat):
    """
    Calculate the mean deflection of the given matrix.
    """
    mat_coo = mat.tocoo()
    nside = hpt.npix2nside(mat_coo.shape[0])
    ang = hpt.angle(nside, mat_coo.row, mat_coo.col)
    return sum(mat_coo.data * ang) / sum(mat_coo.data)


def extragalactic_vector(mat, i):
    """
    Return the HEALpix vector of extragalactic directions
    for a given matrix and observed pixel i.
    """
    row = mat.getrow(i)
    return np.array(row.todense())[0]


def observed_vector(mat, j):
    """
    Return the HEALpix vector of observed directions
    for a given matrix and extragalactic pixel j.
    """
    col = mat.getcol(j)
    return np.array(col.transpose().todense())[0]


def transform_pix_mean(lens, j):
    """
    Transform a galactic direction to the mean observed direction
    Returns the transformed x, y, z, the total probability and the 68% opening angle
    """
    v = observed_vector(lens, j)
    vp = np.sum(v)

    if vp == 0:
        x, y, z = hpt.pix2vec(lens.nside, j)
        return x, y, z, 0, 0

    vx, vy, vz = hpt.pix2vec(lens.nside, range(len(v)))

    # calculate mean vector
    mx, my, mz = np.sum(vx * v), np.sum(vy * v), np.sum(vz * v)
    ms = (mx ** 2 + my ** 2 + mz ** 2) ** 0.5
    mx /= ms
    my /= ms
    mz /= ms

    # calculate sigma
    alpha = np.arccos(vx * mx + vy * my + vz * mz)
    v /= vp
    srt = np.argsort(alpha)
    v, vx, vy, vz = v[srt], vx[srt], vy[srt], vz[srt]
    v = np.cumsum(v)
    # noinspection PyTypeChecker
    i = np.searchsorted(v, 0.68)
    a = np.arccos(mx * vx[i - 1] + my * vy[i - 1] + mz * vz[i - 1])

    return mx, my, mz, a, vp


def transform_vec_mean(lens, x, y, z):
    """
    Transform a galactic direction to the mean observed direction
    Returns the transformed x, y, z, the total probability and the 68% opening angle
    """
    j = hpt.vec2pix(lens.nside, x, y, z)
    return transform_pix_mean(lens, j)


class Lens:
    """
    Galactic magnetic field lens class with the following conventions:

     - the lens maps directions at the galactic border (pointing outwards back to the source) to observed directions
       on Earth (pointing outwards)
     - the Galactic coordinate system is used
     - spherical coordinates are avoided
     - for each logarithmic energy bin there is a lens part represented by a matrix
     - energies are given in log10(energy[eV])
     - the matrices (lens_parts) are in compressed sparse column format (scipy.sparse.csc)
     - for each matrix M_ij
        - the row number i indexes the observed direction
        - the column number j the direction at the Galactic edge
     - indices are HEALPix pixel in ring scheme.
    """

    def __init__(self, cfname=None):
        """
        Load and normalize a lens from the given configuration file.
        Otherwise an empty lens is created. Per default load the lens parts on demand
        """
        self.lens_parts = []  # list of matrices in order of ascending energy
        self.lens_paths = []  # list of pathes in order of ascending energy
        self.log10r_mins = []  # lower rigidity bounds per lens (log10(E/Z/[eV]))
        self.log10r_max = 0  # upper rigidity bound of last lens (log10(E/Z/[eV]))
        self.nside = None  # HEALpix nside parameter
        self.neutral_lens_part = None  # matrix for neutral particles
        self.max_column_sum = None  # maximum of column sums of all matrices
        self.load(cfname)

    def load(self, cfname):
        """
        Load and configure the lens from a config file
        filename minR maxR ... in order of ascending rigidity
        """
        self.cfname = cfname
        # noinspection PyTypeChecker
        if not isinstance(cfname, basestring):
            return
        dirname = os.path.dirname(cfname)

        # read cfg header, to find nside
        with open(cfname) as f:
            for line in f:
                if 'nside' in line:
                    nside = int(line[1:].split()[2])
                    # sanity check
                    if hpt.isnsideok(nside):
                        self.nside = nside
                    break

        try:
            dtype = [('fname', 'S1000'), ('lR0', 'f'), ('lR1', 'f'), ('tol', 'f'), ('MCS', 'f')]
            data = np.genfromtxt(cfname, dtype=dtype)
        except ValueError:
            # Except old lens config format
            dtype = [('fname', 'S1000'), ('lR0', 'f'), ('lR1', 'f')]
            data = np.genfromtxt(cfname, dtype=dtype)

        data.sort(order="lR0")
        self.log10r_mins = data["lR0"]
        self.log10r_max = max(data["lR1"])
        if "MCS" in data.dtype.names:
            self.max_column_sum = data["MCS"]
        self.lens_paths = [os.path.join(dirname, fname.decode('utf-8')) for fname in data["fname"]]
        self.lens_parts = self.lens_paths[:]
        self.neutral_lens_part = sparse.identity(hpt.nside2npix(self.nside), format='csc')

    def check_lens_part(self, lp):
        """
        Perform sanity checks and set HEALpix nside parameter.
        """
        nrows, ncols = lp.get_shape()
        if nrows != ncols:
            raise Exception("Matrix not square %i x %i" % (nrows, ncols))
        nside = hpt.npix2nside(nrows)
        if self.nside is None:
            self.nside = nside
        elif self.nside != int(nside):
            raise Exception("Matrix have different HEALpix schemes")

    def get_lens_part(self, log10e, z=1, cache=True):
        """
        Return the matrix corresponding to a given energy log10e [log_10(energy[eV])] and charge number Z

        :param log10e: energy in units log_10(energy / eV) of the lens part
        :param z: charge number z of the lens part
        :param cache: Caches all the loaded lens parts (increases speed, but may consume a lot of memory!)
        :return: the specified lens part
        """
        if z == 0:
            return self.neutral_lens_part
        if not self.lens_parts:
            raise Exception("Lens empty")
        log_rig = log10e - np.log10(z)
        if (log_rig < self.log10r_mins[0]) or (log_rig > self.log10r_max):
            raise ValueError("Rigidity 10^(%.2f - np.log10(%i)) not covered" % (log10e, z))
        i = bisect_left(self.log10r_mins, log_rig) - 1

        if cache:
            if not isinstance(self.lens_parts[i], sparse.csc.csc_matrix):
                lp = load_lens_part(self.lens_parts[i])
                self.check_lens_part(lp)
                self.lens_parts[i] = lp
            return self.lens_parts[i]

        return load_lens_part(self.lens_paths[i])


def apply_exposure_to_lens(lens, a0=-35.25, zmax=60):
    """
    Apply a given exposure (coverage) to all matrices of a lens.

    :param lens: object from class Lens(), which specifies the lens
    :param a0: equatorial declination [deg] of the experiment (default: AUGER, a0=-35.25 deg)
    :param zmax: maximum zenith angle [deg] for the events
    """
    coverage = hpt.exposure_pdf(lens.nside, a0, zmax)
    lens.multiply_diagonal_matrix(coverage)


def plot_col_sum(mat):
    """plots the sum of all columns"""
    col_sums = mat.sum(axis=0).tolist()[0]
    plt.plot(col_sums, c='b', lw=0.5)


def plot_row_sum(mat):
    """plots the sum of all rows"""
    row_sums = mat.sum(axis=1).tolist()
    plt.plot(row_sums, c='r', lw=0.5)


def plot_matrix(mat_csc, stride=100):
    """Plots a CSC matrix as scatterplot"""
    mat = mat_csc.tocoo()
    plt.figure()
    plt.scatter(mat.col[::stride], mat.row[::stride], marker='+')
    plt.xlim(0, mat.shape[0])
    plt.ylim(0, mat.shape[1])
    ax = plt.gca()
    ax.invert_yaxis()
    ax.set_xticks(())
    ax.set_yticks(())
