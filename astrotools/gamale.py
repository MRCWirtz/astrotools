"""
Galactic magnetic field lens
see PARSEC: A Parametrized Simulation Engine for Ultra-High Energy Cosmic Ray Protons, arXiv:1302.3761
http://web.physik.rwth-aachen.de/Auger_MagneticFields/PARSEC/
"""
import gzip
import os
from struct import pack, unpack

import numpy as np
from scipy import sparse

import astrotools.healpytools as hpt

# python 2/3 compatibility
try:
    basestring
except NameError:
    basestring = str  # pylint: disable=W0622,C0103


def save_lens_part(mat, fname):
    """
    Save the lens part in PARSEC format (coordinate type sparse format).
    """
    if fname.endswith(".npz"):
        if not isinstance(mat, sparse.csc_matrix):
            try:  # this works e.g. for scipy.sparse.lil_matrix
                mat = mat.tocsc()
            except AttributeError:
                raise AttributeError("Data can not be converted into csc format")
        np.savez(fname, data=mat.data, indices=mat.indices, indptr=mat.indptr, shape=mat.shape)
    else:
        if not isinstance(mat, sparse.coo_matrix):
            try:  # this works e.g. for scipy.sparse.lil_matrix
                mat = mat.tocoo()
            except AttributeError:
                raise AttributeError("Data can not be converted into csc format")
        fout = open(fname, 'wb')
        fout.write(pack('i', mat.nnz))
        fout.write(pack('i', mat.shape[0]))
        fout.write(pack('i', mat.shape[1]))
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
    if fname.endswith(".npz"):
        data = np.load(fname)
        return sparse.csc_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
    else:
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


def mat2nside(mat):
    """
    Calculate nside from a given lenspart matrice.
    """
    nrows, ncols = mat.get_shape()
    if nrows != ncols:
        raise Exception("Matrix not square %i x %i" % (nrows, ncols))
    nside = hpt.npix2nside(nrows)
    return nside


def extragalactic_vector(mat, i):
    """
    Return the HEALpix vector of extragalactic directions
    for a given matrix and observed pixel i.
    """
    row = mat.getrow(i)
    return np.array(row.todense())[0].astype(float)


def observed_vector(mat, j):
    """
    Return the HEALpix vector of observed directions
    for a given matrix and extragalactic pixel j.
    """
    col = mat.getcol(j)
    return np.array(col.transpose().todense())[0].astype(float)


def mean_deflection(mat):
    """
    Calculate the mean deflection of the given matrix.
    """
    mat_coo = mat.tocoo()
    nside = hpt.npix2nside(mat_coo.shape[0])
    ang = hpt.angle(nside, mat_coo.row, mat_coo.col)
    return sum(mat_coo.data * ang) / sum(mat_coo.data)


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
        self.lens_parts = []    # list of matrices in order of ascending energy
        self.lens_paths = []    # list of pathes in order of ascending energy
        self.log10r_mins = []   # lower rigidity bounds of lens (log10(E/Z/[eV]))
        self.log10r_max = []    # upper rigidity bounds of lens (log10(E/Z/[eV]))
        self.dlog10r = None     # rigidity bin width
        self.nside = None       # HEALpix nside parameter
        self.stat = None
        self.neutral_lens_part = None   # matrix for neutral particles
        self.max_column_sum = None      # maximum of column sums of all matrices
        self.cfname = cfname
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
                    self.nside = int(line[1:].split()[2])
                    # sanity check
                    assert hpt.isnsideok(self.nside), "Healpy nside value from .cfg header not OK."
                if 'stat' in line:
                    self.stat = int(line[1:].split()[2])
                if ('.npz' in line) or ('.mldat' in line):
                    break
        try:
            dtype = [('fname', 'S1000'), ('lR0', float), ('lR1', float), ('tol', float), ('MCS', float)]
            data = np.genfromtxt(cfname, dtype=dtype)
            self.max_column_sum = data["MCS"]
            self.tolerance = data["tol"]
        except ValueError:  # pragma: no cover
            # Except old lens config format
            dtype = [('fname', 'S1000'), ('lR0', float), ('lR1', float)]
            data = np.genfromtxt(cfname, dtype=dtype)

        data.sort(order="lR0")
        self.log10r_mins = data["lR0"]
        self.log10r_max = data["lR1"]
        self.dlog10r = (data["lR1"][0] - data["lR0"][0]) / 2.
        assert np.allclose(data["lR1"], data["lR0"] + 2 * self.dlog10r)
        self.lens_paths = [os.path.join(dirname, fname.decode('utf-8')) for fname in data["fname"]]
        self.lens_parts = self.lens_paths[:]    # Fill with matrices first when is neeed
        self.neutral_lens_part = sparse.identity(hpt.nside2npix(self.nside), format='csc')

    def check_lens_part(self, lp):
        """
        Perform sanity checks and set HEALpix nside parameter.
        """
        nside = mat2nside(lp)
        if self.nside is None:
            self.nside = nside
        elif self.nside != nside:
            raise Exception("Matrix have different HEALpix nside than in .cfg header")

        stat = int(lp.sum(axis=1).max())
        if self.stat is None:
            self.stat = stat
        elif self.stat != stat:
            raise Exception("Matrix have different stat than in .cfg header")

        return True

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
        log10r = log10e - np.log10(z)
        log10r_bins = np.append(self.log10r_mins, np.max(self.log10r_max))
        i = np.digitize(log10r, log10r_bins) -1
        is_i_in_limits = (i < 0) or (i < len(log10r_bins) - 1)
        if is_i_in_limits:
            diff2bin = np.abs(self.log10r_mins[i] + self.dlog10r - log10r)
            is_close = np.isclose(max(self.dlog10r, diff2bin), self.dlog10r)
        else:
            is_close = False
        if not is_i_in_limits or not is_close:
            raise ValueError("Rigidity 10^(%.2f - np.log10(%i)) not covered" % (log10r, z))
        if isinstance(self.lens_parts[i], sparse.csc.csc_matrix):
            return self.lens_parts[i]
        lp = load_lens_part(self.lens_paths[i])
        self.check_lens_part(lp)
        if cache:
            self.lens_parts[i] = lp
            return lp

        return lp
