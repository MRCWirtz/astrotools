"""
Tools for CRPropa 3
"""
import numpy as np


def nucleus2id(A, Z):
    """
    Given a mass and charge number, returns the nucleus ID (2006 PDG standard).

    :param A: mass number
    :param Z: charge number
    :return nucleus ID
    """
    return 1000000000 + Z * 10000 + A * 10


def id2z(pid):
    """
    Given a nucleus ID (2006 PDG standard), returns the charge number.

    :param pid: nucleus ID
    :return charge number
    """
    return pid % 1000000000 // 10000


def id2a(pid):
    """
    Given a nucleus ID (2006 PDG standard), returns the mass number.

    :param pid: nucleus ID
    :return mass number
    """
    return pid % 10000 // 10


class Charge2Mass:
    """
    Convert the charge of a cosmic ray to it's mass by different assumptions
    """
    def __init__(self, charge):
        self.int = isinstance(charge, int)
        self.charge = np.array([charge]) if self.int else np.array(charge)

    def double(self):
        """
        Simple approach of A = 2 * Z
        """
        # A = 2 * Z
        A = 2 * self.charge.astype(np.int)
        A[A <= 2] = 1           # For H, probably A=1

        return A[0] if self.int else A

    def empiric(self):
        """
        A = Z * (2 + a_c / (2 a_a) * A**(2/3))    [https: // en.wikipedia.org / wiki / Semi - empirical_mass_formula]
        with a_c = 0.714 and aa = 23.2
        Inverse approximation: A(Z) = 2 * Z + a * Z**b with a=0.0200 and b = 1.748
        """
        a = 0.02
        b = 1.748
        A = np.rint(2 * self.charge + a * self.charge ** b).astype(np.int)
        A[A <= 2] = 1           # For H, probably A=1

        return A[0] if self.int else A

    def stable(self):
        """
        Using uniform distribution within all stable mass numbers of a certain charge number
        """
        stable = {1: [1], 2: [3, 4], 3: [6, 7], 4: [9], 5: [10, 11], 6: [12, 13], 7: [14, 15], 8: [16, 17, 18], 9: [19],
                  10: [20, 21, 22], 11: [23], 12: [24, 25, 26], 13: [27], 14: [28, 29, 30], 15: [31], 16: [32, 33, 34, 36],
                  17: [35, 37], 18: [36, 38, 40], 19: [39, 40, 41], 20: [40, 42, 43, 44, 46, 48], 21: [45],
                  22: [46, 47, 48, 49, 50], 23: [51], 24: [50, 52, 53, 54], 25: [55], 26: [54, 56, 57, 58], 27: [59]}

        Z = np.array(self.charge)
        A = np.zeros(Z.shape).astype(np.int)
        for zi in np.unique(Z):
            mask = Z == zi
            p = np.ones(len(stable[zi])) / len(stable[zi])
            A[mask] = np.random.choice(stable[zi], size=np.sum(mask), p=p).astype(np.int)

        return A[0] if self.int else A

    def abundance(self):
        # use abundance in our solar system (milky way?)
        # TODO
        return None


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
