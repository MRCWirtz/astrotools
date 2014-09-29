# Tools for CRPropa 3
import numpy as np


def nucleusID(A, Z):
    """ Given a mass and charge number, returns the nucleus ID (2006 PDG standard). """
    return 1000000000 + Z * 10000 + A * 10

def nucleusID2Z(pid):
    """ Given a nucleus ID (2006 PDG standard), returns the charge number. """
    return pid % 1000000000 // 10000

def nucleusID2A(pid):
    """ Given a nucleus ID (2006 PDG standard), returns the mass number. """
    return pid % 10000 // 10

def iter_loadtxt(filename, delimiter='\t', skiprows=0, dtype=float, unpack=False):
    """
    Lightweight loading function for large tabulated data files in ASCII format.
    Memory requirement is greatly reduced compared to np.genfromtxt and np.loadtxt.
    """
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    if unpack:
        return data.transpose()
    return data
