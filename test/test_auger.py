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
np.random.seed(0)


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
        x_gumbel = auger.rand_xmax(np.ones(stat) * log10e_cens[i], np.ones(stat) * A[i], model=model)
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

    def test_05_xmax_resolution(self):

        log10e = 19.
        res700 = auger.xmax_resolution(700, log10e)      # likely heavy
        res750 = auger.xmax_resolution(750, log10e)      # medium
        res850 = auger.xmax_resolution(850, log10e)    # likely light
        self.assertTrue(res700 > 0)
        self.assertTrue(res750 > 0)
        self.assertTrue(res850 > 0)

    def test_06_xmax_acceptance(self):
        log10e = 19.
        mass = 4
        xmax = auger.rand_xmax(log10e, mass, 10)
        acceptance = auger.xmax_acceptance(xmax, log10e)
        self.assertTrue((acceptance == 1.).all())
        acceptance_lo = auger.xmax_acceptance(0.5 * xmax, log10e)
        self.assertTrue((acceptance_lo < 1.).all())
        acceptance_hi = auger.xmax_acceptance(1.5 * xmax, log10e)
        self.assertTrue((acceptance_hi < 1.).all())

    def test_07_lna_xmax_moments(self):

        n = 100000
        log10e = auger.rand_energy_from_auger(n, log10e_min=17.8)
        masses = 2 * auger.rand_charge_from_auger(log10e)
        l_ec, mln_a, vln_a = auger.ln_a_moments(log10e, masses)
        # check if uhecrs get heavier
        self.assertTrue((mln_a[0] < mln_a[-1]) & (vln_a[0] > vln_a[-1]))

        m_xmax, v_xmax = auger.ln_a_moments2xmax_moments(l_ec, mln_a, vln_a)
        std_xmax = np.sqrt(auger.var_xmax(l_ec, 0.8*np.exp(mln_a)))
        self.assertTrue((std_xmax < 100).all())

        mln_a2, vln_a2 = auger.xmax_moments2ln_a_moments(l_ec, m_xmax, v_xmax)
        self.assertTrue(np.allclose(mln_a, mln_a2))
        self.assertTrue(np.allclose(vln_a, vln_a2))


class EnergyCharge(unittest.TestCase):

    def test_01_composition(self):

        n = 1000
        log10e = 18.5 + np.random.random(n)
        charge = auger.rand_charge_from_auger(log10e)

        self.assertTrue(charge.size == n)
        self.assertTrue((charge >= 1).all() & (charge <= 26).all())

    def test_02_charge_fit_from_auger(self):

        fractions_18 = auger.charge_fit_from_auger(18.)
        self.assertAlmostEqual(np.sum(fractions_18), 1.)
        fractions_19 = auger.charge_fit_from_auger(19.)
        self.assertAlmostEqual(np.sum(fractions_19), 1.)

        # check physics (more proton at lower energies)
        self.assertTrue(fractions_18[0] > fractions_19[0])
        self.assertTrue(fractions_18[-1] < fractions_19[-1])

    def test_03_energy(self):

        n = 1000
        log10e_min = 19.
        log10e = auger.rand_energy_from_auger(n, log10e_min)
        self.assertTrue(log10e.size == n)
        self.assertTrue((log10e >= log10e_min).all() & (log10e <= 20.5).all())
        self.assertTrue(len(log10e[log10e > log10e_min + 0.1]) < n)


class Gumbel(unittest.TestCase):

    def test_01_gumbel_pdf(self):

        xmax = np.linspace(600, 1100, 500)
        n = 1000
        pdf_p = auger.gumbel(xmax, 19., 1)
        pdf_he = auger.gumbel(xmax, 19., 4)
        pdf_cno = auger.gumbel(xmax, 19., 16)
        pdf_fe = auger.gumbel(xmax, 19., 56)
        self.assertTrue(np.argmax(pdf_p) > np.argmax(pdf_he))
        self.assertTrue(np.argmax(pdf_he) > np.argmax(pdf_cno))
        self.assertTrue(np.argmax(pdf_cno) > np.argmax(pdf_fe))

        xmax_p = np.random.choice(xmax, n, p=pdf_p/pdf_p.sum())
        xmax_he = np.random.choice(xmax, n, p=pdf_he/pdf_he.sum())
        xmax_cno = np.random.choice(xmax, n, p=pdf_cno/pdf_cno.sum())
        xmax_fe = np.random.choice(xmax, n, p=pdf_fe/pdf_fe.sum())

        self.assertTrue(np.std(xmax_p) > np.std(xmax_he))
        self.assertTrue(np.std(xmax_he) > np.std(xmax_cno))
        self.assertTrue(np.std(xmax_cno) > np.std(xmax_fe))

    def test_02_gumbel_cdf_sf(self):

        xmax = np.linspace(600, 1100, 1000)
        pdf = auger.gumbel(xmax, 19., 1)
        cdf_pdf = np.cumsum(pdf)
        cdf_pdf /= cdf_pdf[-1]
        cdf = auger.gumbel_cdf(xmax, 19., 1)
        cdf /= cdf[-1]
        sf = auger.gumbel_sf(xmax, 19., 1)
        sf /= sf[0]
        self.assertTrue(np.all(np.abs(cdf_pdf[:-1] - cdf[:-1]) < 5e-3))
        self.assertTrue(np.all(np.abs(cdf[:-1] + sf[:-1] - 1) < 5e-3))


if __name__ == '__main__':
    unittest.main()
