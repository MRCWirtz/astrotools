"""
Tools for calculating nuclei properties
"""
import numpy as np


def nucleus2id(mass, charge):
    """
    Given a mass and charge number, returns the nucleus ID (2006 PDG standard).

    :param mass: mass number
    :param charge: charge number
    :return: nucleus ID
    """
    return 1000000000 + charge * 10000 + mass * 10


def id2z(pid):
    """
    Given a nucleus ID (2006 PDG standard), returns the charge number.

    :param pid: nucleus ID
    :return: charge number
    """
    return pid % 1000000000 // 10000


def id2a(pid):
    """
    Given a nucleus ID (2006 PDG standard), returns the mass number.

    :param pid: nucleus ID
    :return: mass number
    """
    return pid % 10000 // 10


class Charge2Mass:
    """
    Convert the charge of a cosmic ray to it's mass by different assumptions
    """
    def __init__(self, charge):
        self.scalar = isinstance(charge, (int, float))
        charge = np.array([charge]) if self.scalar else np.array(charge)
        self.type = charge.dtype
        self.charge = charge

    def double(self):
        """
        Simple approach of mass = 2 * Z
        """
        # mass = 2 * Z
        mass = 2. * self.charge
        mass[mass <= 2] = 1           # For H, probably mass=1
        return self._return(mass)

    def empiric(self):
        """
        A = Z * (2 + a_c / (2 a_a) * A**(2/3))    [https: // en.wikipedia.org / wiki / Semi - empirical_mass_formula]
        with a_c = 0.714 and aa = 23.2
        Inverse approximation: A(Z) = 2 * Z + a * Z**b with a=0.0200 and b = 1.748
        """
        a = 0.02
        b = 1.748
        mass = 2. * self.charge + a * self.charge ** b
        mass[mass <= 2] = 1           # For H, probably A=1
        return self._return(mass)

    def stable(self):
        """
        Using uniform distribution within all stable mass numbers of a certain charge number
        """
        stable = {1: [1], 2: [3, 4], 3: [6, 7], 4: [9], 5: [10, 11], 6: [12, 13], 7: [14, 15], 8: [16, 17, 18], 9: [19],
                  10: [20, 21, 22], 11: [23], 12: [24, 25, 26], 13: [27], 14: [28, 29, 30], 15: [31],
                  16: [32, 33, 34, 36], 17: [35, 37], 18: [36, 38, 40], 19: [39, 40, 41], 20: [40, 42, 43, 44, 46, 48],
                  21: [45], 22: [46, 47, 48, 49, 50], 23: [51], 24: [50, 52, 53, 54], 25: [55],
                  26: [54, 56, 57, 58], 27: [59]}

        if self.type != int:
            raise TypeError("Expected int input for stable charge converter!")

        charge = np.array(self.charge)
        mass = np.zeros(charge.shape).astype(np.int)
        for zi in np.unique(charge):
            mask = charge == zi
            p = np.ones(len(stable[zi])) / len(stable[zi])
            mass[mask] = np.random.choice(stable[zi], size=np.sum(mask), p=p).astype(np.int)
        return self._return(mass)

    def _return(self, mass):
        if self.type == int:
            mass = np.rint(mass).astype(int)

        return mass[0] if self.scalar else mass


class Mass2Charge:
    """
    Convert the mass of a cosmic ray to it's charge by different assumptions
    """
    def __init__(self, mass):
        self.scalar = isinstance(mass, (int, float))
        mass = np.array([mass]) if self.scalar else np.array(mass)
        self.type = mass.dtype
        self.mass = mass

    def double(self):
        """
        Simple approach of charge: Z = A / 2
        """
        # Z = A / 2
        charge = self.mass / 2.
        charge[charge < 1] = 1           # Minimum is 1
        return self._return(charge)

    def empiric(self):
        """
        A = Z * (2 + a_c / (2 a_a) * A**(2/3))
        [https: // en.wikipedia.org / wiki / Semi - empirical_mass_formula]
        with a_c = 0.714 and a_a = 23.2
        """
        a_c = 0.714
        a_a = 23.2
        charge = self.mass / (2 + a_c / (2 * a_a) * self.mass**(2 / 3.))
        charge[charge < 1] = 1           # Minimum is 1
        return self._return(charge)

    def stable(self):
        """
        Using uniform distribution within all possible stable charges of a certain mass number
        (can not deal with float inputs and unstable A = 5, 8)
        """
        stable = {1: [1], 2: [1], 3: [2], 4: [2], 6: [3], 7: [3], 9: [4], 10: [5], 11: [5], 12: [6], 13: [6], 14: [7],
                  15: [7], 16: [8], 17: [8], 18: [8], 19: [9], 20: [10], 21: [10], 22: [10], 23: [11], 24: [12],
                  25: [12], 26: [12], 27: [13], 28: [14], 29: [14], 30: [14], 31: [15], 32: [16], 33: [16], 34: [16],
                  35: [17], 36: [16, 18], 37: [17], 38: [18], 39: [19], 40: [18, 19, 20], 41: [19], 42: [20], 43: [20],
                  44: [20], 45: [21], 46: [20, 22], 47: [22], 48: [20, 22], 49: [22], 50: [22, 24], 51: [23], 52: [24],
                  53: [24], 54: [24, 26], 55: [25], 56: [26], 57: [26], 58: [26], 59: [27]}
        if self.type != int:
            raise TypeError("Expected int input for stable charge converter!")
        elif 5 in self.mass:
            raise KeyError("A=5 excluded because no stable nucleus exists!")
        elif 8 in self.mass:
            raise KeyError("A=8 excluded because no stable nucleus exists!")

        mass = np.array(self.mass)
        charge = np.zeros(mass.shape).astype(np.int)
        for mi in np.unique(mass):
            mask = mass == mi
            p = np.ones(len(stable[mi])) / len(stable[mi])
            charge[mask] = np.random.choice(stable[mi], size=np.sum(mask), p=p).astype(np.int)
        return self._return(charge)

    def _return(self, charge):
        if self.type == int:
            charge = np.rint(charge).astype(int)

        return charge[0] if self.scalar else charge


def iter_loadtxt(filename, delimiter='\t', skiprows=0, dtype=float, unpack=False):  # pragma: no cover
    """
    Lightweight loading function for large tabulated data files in ASCII format.
    Memory requirement is greatly reduced compared to np.genfromtxt and np.loadtxt.
    """
    def iter_func():
        """helper function"""
        line = ""
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
