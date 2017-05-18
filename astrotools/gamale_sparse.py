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
    log_rig = np.append(lens.lRmins, lens.lRmax)
    name = ".".join(os.path.basename(lens.cfname).split(".")[:-1])
    # write the new config file
    cname = os.path.join(outdir, name + ".cfg")
    cfile = file(cname, "w")
    cfile.write("# fname logEmin logEmax maxColumnSum\n")
    cfile.write("# nside = %i\n" % lens.nside)

    for i, lp in enumerate(lens.lensParts):
        _lp = gamale.load_lens_part(lp)
        part_name = "%s_%i.npz" % (name, i)
        outpath = os.path.join(outdir, part_name)
        save_data(outpath, _lp)
        cfile.write("%s %.6f %.6f\n" % (part_name, log_rig[i], log_rig[i + 1]))
    cfile.close()
    return cname


def load_sparse_lens_part(lp):
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
     - the matrices (lensParts) are in compressed sparse column format (scipy.sparse.csc)
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
        self.lensParts = []  # list of matrices in order of ascending energy
        self.lensPaths = []  # list of pathes in order of ascending energy
        self.lRmins = []  # lower rigidity bounds per lens (log10(E/Z/[eV]))
        self.lRmax = 0  # upper rigidity bound of last lens (log10(E/Z/[eV]))
        self.nside = None  # HEALpix nside parameter
        self.neutralLensPart = None  # matrix for neutral particles
        self.maxColumnSum = None  # maximum of column sums of all matrices
        self.cfname = cfname
        if cfname is not None:
            self.load(cfname)
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

        # read cfg header, to find nside
        with open(cfname) as f:
            for line in f:
                # only read the inital comments
                if not line.startswith("#"):
                    break

                parts = line[1:].split()
                if len(parts) > 2:
                    if parts[0] == "nside":
                        nside = int(parts[2])
                        # sanity check
                        if nside < 0 or nside > 1000000:
                            self.nside = None
                        else:
                            self.nside = nside
        data = np.genfromtxt(cfname, dtype=[('fname', 'S1000'), ('lR0', 'f'), ('lR1', 'f')])

        data.sort(order="lR0")
        self.lRmins = data["lR0"]
        self.lRmax = max(data["lR1"])
        self.lensPaths = [os.path.join(dirname, fname) for fname in data["fname"]]
        self.lensParts = self.lensPaths[:]
        self.neutralLensPart = sparse.identity(hpt.nside2npix(self.nside), format='csc')

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
            return self.neutralLensPart
        if len(self.lensParts) == 0:
            raise Exception("Lens empty")
        log_rig = log10e - np.log10(z)
        if (log_rig < self.lRmins[0]) or (log_rig > self.lRmax):
            raise ValueError("Rigidity %f/%i EeV not covered" % (log10e, z))
        i = bisect_left(self.lRmins, log_rig) - 1

        if cache:
            if not isinstance(self.lensParts[i], sparse.csc.csc_matrix):
                lp = load_sparse_lens_part(self.lensParts[i])
                self.check_lens_part(lp)
                self.lensParts[i] = lp
            return self.lensParts[i]

        return load_sparse_lens_part(self.lensPaths[i])
