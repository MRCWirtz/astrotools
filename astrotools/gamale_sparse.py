"""
Sparse implementation of Galactic magnetic field lens
see PARSEC: A Parametrized Simulation Engine for Ultra-High Energy Cosmic Ray Protons, arXiv:1302.3761
http://web.physik.rwth-aachen.de/Auger_MagneticFields/PARSEC/
"""
from bisect import bisect_left
import os

import numpy as np
from scipy import sparse

from astrotools import gamale, healpytools as hpt

__author__ = 'Martin Urban'

# python 2/3 compatibility
try:
    unicode
except NameError:
    unicode = str  # pylint: disable=W0622,C0103
try:
    basestring
except NameError:
    basestring = str  # pylint: disable=W0622,C0103


def save_data(outpath, data):
    """
    A function to save data, up to now only output via numpy is implemented, it is planned to do implement a hd5 output

    :param outpath: name of output file
    :param data: data, e.g. a numpy array or a list or tuple or even an object which can be pickled
    """
    if not isinstance(data, (sparse.csr_matrix, sparse.csc_matrix)):
        try:  # this works e.g. for scipy.sparse.lil_matrix
            data = data.tocsc()
        except AttributeError:
            raise AttributeError("Data can not be converted into csc format")
    outpath = outpath if outpath.endswith(".npz") else outpath + ".npz"
    np.savez(outpath, data=data.data, indices=data.indices, indptr=data.indptr,
             shape=data.shape)


def load_data(outpath):
    """
    A function that tries to generalize the loading of data files

    :param outpath: name of the file
    :return: output e.g. in form of a numpy array
    """
    filepath = outpath if outpath.endswith(".npz") else outpath + ".npz"
    data = np.load(filepath)
    return sparse.csc_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])


def convert_from_gamale_to_sparse(l, outdir):
    """
    Conversion function from gamale to gamale sparse format

    :param l: gamale.lens instance or str
    :param outdir: output directory, filenames are detected automatically
    :return: name of the new config file
    """
    if os.path.isfile(outdir):
        raise NameError("The outputfile can not created because it exists already as file: %s" % outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    lens = gamale.Lens(l) if isinstance(l, (str, unicode)) else l
    log_rig = np.append(lens.log10r_mins, lens.log10r_max)
    name = ".".join(os.path.basename(lens.cfname).split(".")[:-1])
    # write the new config file
    cname = os.path.join(outdir, name + ".cfg")
    cfile = open(cname, "w")
    cfile.write("# fname logEmin logEmax max_column_sum\n")
    cfile.write("# nside = %i\n" % lens.nside)

    for i, lp in enumerate(lens.lens_parts):
        _lp = gamale.load_lens_part(lp)
        part_name = "%s_%i.npz" % (name, i)
        outpath = os.path.join(outdir, part_name)
        save_data(outpath, _lp)
        cfile.write("%s %.6f %.6f\n" % (part_name, log_rig[i], log_rig[i + 1]))
    cfile.close()
    return cname


def load_sparse_lens_part(lp):
    """equivalent to :meth:`gamale_sparse.load_data`"""
    return load_data(lp)


class SparseLens:
    """
    Galactic magnetic field lens class duplicate for sparse lenses with the following conventions:

     - the lens maps directions at the galactic border (pointing outwards back to the source)
       to observed directions on Earth (pointing outwards)
     - the Galactic coordinate system is used
     - spherical coordinates are avoided
     - for each logarithmic energy bin there is a lens part represented by a matrix
     - energies are given in EeV
     - the matrices (lens_parts) are in compressed sparse column format (scipy.sparse.csc)
     - for each matrix M_ij
        - the row number i indexes the observed direction
        - the column number j the direction at the Galactic edge
     - indices are HEALpixel in ring scheme.
    """

    def __init__(self, cfname=None, gamale_lens=None, outdir="/tmp/sparse_lens"):
        """
        Load and normalize a lens from the given configuration file or a given gamale_lens
        Otherwise an empty lens is created. Load the lens parts on demand

        :param cfname: config file name
        :param gamale_lens: gamale lens
        :param outdir: output dir where to save the converted lens
        """
        self.lens_parts = []  # list of matrices in order of ascending energy
        self.lens_paths = []  # list of pathes in order of ascending energy
        self.log10r_mins = []   # lower rigidity bounds of lens (log10(E/Z/[eV]))
        self.log10r_max = []    # upper rigidity bounds of lens (log10(E/Z/[eV]))
        self.dlog10e = None
        self.nside = None  # HEALpix nside parameter
        self.neutral_lens_part = None  # matrix for neutral particles
        self.max_column_sum = None  # maximum of column sums of all matrices
        self.cfname = cfname
        if gamale_lens is not None:
            self.cfname = convert_from_gamale_to_sparse(gamale_lens, outdir=outdir)
        self.load(self.cfname)

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
        self.log10r_max = data["lR1"]
        self.dlog10e = (data["lR1"][0] - data["lR0"][0]) / 2.
        assert np.array_equal(data["lR1"], data["lR0"] + 2 * self.dlog10e)
        if "MCS" in data.dtype.names:
            self.max_column_sum = data["MCS"]
        self.lens_paths = [os.path.join(dirname, fname.decode('utf-8')) for fname in data["fname"]]
        self.lens_parts = self.lens_paths[:]    # Fill with matrices first when is neeed
        self.neutral_lens_part = sparse.identity(hpt.nside2npix(self.nside), format='csc')

    def check_lens_part(self, lp):
        """
        Perform sanity checks and set HEALpix nside parameter.

        :param lp: lenspart
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
        Return the matrix corresponding to a given energy (in log10(E / eV)) and charge number z

        :param log10e: energy in log_10(energy / eV), e.g. 18.25
        :param z: charge number of the lens part
        :return: sparse csc_matrix
        """
        if z == 0:
            return self.neutral_lens_part
        if not self.lens_parts:
            raise Exception("Lens empty")
        log_rig = log10e - np.log10(z)
        if np.min(np.abs(self.log10r_mins + self.dlog10e - log_rig)) > self.dlog10e:
            raise ValueError("Rigidity %f/%i EeV not covered" % (log10e, z))
        i = bisect_left(self.log10r_mins, log_rig) - 1

        if isinstance(self.lens_parts[i], sparse.csc.csc_matrix):
            return self.lens_parts[i]
        elif cache:
            lp = load_sparse_lens_part(self.lens_parts[i])
            self.check_lens_part(lp)
            self.lens_parts[i] = lp
            return lp

        return load_sparse_lens_part(self.lens_paths[i])
