import unittest
from astrotools import auger
import matplotlib.pyplot as plt
import numpy as np

nlog10e = 20
stat = 50000
log10e_bins = np.linspace(18, 20, nlog10e + 1)
log10e_cens = (log10e_bins[1:] + log10e_bins[:-1]) / 2
A = np.ones(nlog10e)
plots = False


def plotting(m_x2, s_x2, model):

    l_e, m_x1, v_x1 = auger.xmax_moments(log10e_cens, A, bins=log10e_bins, model=model)

    if not plots:
        return

    fix, ax = plt.subplots(2, 1, figsize=(14, 12))
    ax1, ax2 = ax[0], ax[1]
    ax1.plot(log10e_cens, m_x1, label='lnA -> Xmax')
    ax1.plot(log10e_cens, m_x2, label='gumbel -> Xmax')
    ax1.legend(loc='upper left', frameon=False)
    ax1.set_xlabel('$\log_{10}$(E/[eV])')
    ax1.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g cm$^{-2}$]')

    ax2.plot(log10e_cens, v_x1 ** .5, label='lnA -> Xmax')
    ax2.plot(log10e_cens, s_x2, label='gumbel -> Xmax')
    ax2.legend(loc='upper left', frameon=False)
    ax2.set_xlabel('$\log_{10}$(E/[eV])')
    ax2.set_ylabel(r'$\sigma(\rm{X_{max}})$ [g cm$^{-2}$]')

    plt.suptitle(model)
    plt.tight_layout()
    plt.savefig('xmax_lna_%s.png' % model)
    plt.clf()


def moments(model):

    m_x2 = np.zeros(nlog10e)
    s_x2 = np.zeros(nlog10e)
    for i in range(nlog10e):
        x_gumbel = auger.rand_gumbel(np.ones(stat) * log10e_cens[i], np.ones(stat) * A[i], model=model)
        m_x2[i] = np.mean(x_gumbel)
        s_x2[i] = np.std(x_gumbel)

    return m_x2, s_x2


class TestXmaxlNA(unittest.TestCase):

    def test_01_epos_lhc(self):

        model = 'EPOS-LHC'
        m_x2, s_x2 = moments(model)
        plotting(m_x2, s_x2, model)

        self.assertTrue(True)

    def test_02_sybill(self):

        model = 'Sibyll2.1'
        m_x2, s_x2 = moments(model)
        plotting(m_x2, s_x2, model)

        self.assertTrue(True)

    def test_03_qgs2(self):

        model = 'QGSJetII'
        m_x2, s_x2 = moments(model)
        plotting(m_x2, s_x2, model)

        self.assertTrue(True)

    def test_04_qgs4(self):

        model = 'QGSJetII-04'
        m_x2, s_x2 = moments(model)
        plotting(m_x2, s_x2, model)

        self.assertTrue(True)


class EnergyAndCharge(unittest.TestCase):

    def test_01_composition(self):

        n = 1000
        log10e = 18.5 + np.random.random(n)
        charge = auger.rand_charge_from_auger(log10e)

        self.assertTrue(charge.size == n)
        self.assertTrue((charge >= 1).all() & (charge <= 26).all())

    def test_02_energy(self):

        n = 1000
        log10e_min = 19.
        log10e = auger.rand_energy_from_auger(n, log10e_min)
        self.assertTrue(log10e.size == n)
        self.assertTrue((log10e >= log10e_min).all() & (log10e <= 20.5).all())
        self.assertTrue(len(log10e[log10e > log10e_min + 0.1]) < n)


if __name__ == '__main__':
    unittest.main()
