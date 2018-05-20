"""
Tools related to the Pierre Auger Observatory
"""
from os import path

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.special

import astrotools.stat as stat

# References
# [1] Manlio De Domenico et al., JCAP07(2013)050, doi:10.1088/1475-7516/2013/07/050
# [2] S. Adeyemi and M.O. Ojo, Kragujevac J. Math. 25 (2003) 19-29
# [3] JCAP 1302 (2013) 026, Interpretation of the Depths of Maximum of Extensive Air Showers Measured by the
#     Pierre Auger Observatory, DOI:10.1088/1475-7516/2013/02/026, arXiv: 1301.6637
# [4] Auger ICRC'15, data file received from Ines Valino on 2015-07-30
# [5] GAP-2014-083, Update of the parameterisations given in "Interpretation of the Depths ..."
#     in the energy range 10^17 - 10^20 eV
# [6] Long Xmax paper
# [7] Depth of Maximum of Air-Shower Profiles at the Pierre Auger Observatory: Composition Implications (2014)
#     The Pierre Auger Collaboration, arXiv: 1409.5083

# --------------------- DATA -------------------------
DATA_PATH = path.split(__file__)[0] + '/data'

# Spectrum data [4]
# noinspection PyTypeChecker
DSPECTRUM = np.genfromtxt(
    DATA_PATH + '/auger_spectrum_2015.txt', delimiter=',', names=True)
# from Ines Valino, ICRC2015
DSPECTRUM_ANALYTIC = np.array([3.3e-19, 4.82e18, 42.09e18, 3.29, 2.6, 3.14])

# Xmax data of [6], from http://www.auger.org/data/xmax2014.tar.gz on
# 2014-09-29
# noinspection PyTypeChecker
DXMAX = {
    'histograms': np.genfromtxt(DATA_PATH + '/xmax/xmaxHistograms.txt', usecols=range(7, 107)),
    'moments': np.genfromtxt(DATA_PATH + '/xmax/xmaxMoments.txt', names=True, usecols=range(3, 13)),
    'resolution': np.genfromtxt(DATA_PATH + '/xmax/resolution.txt', names=True, usecols=range(3, 8)),
    'acceptance': np.genfromtxt(DATA_PATH + '/xmax/acceptance.txt', names=True, usecols=range(3, 11)),
    'systematics': np.genfromtxt(DATA_PATH + '/xmax/xmaxSystematics.txt', names=True, usecols=(3, 4)),
    'resolutionNoFOV': np.genfromtxt(DATA_PATH + '/xmax/resolutionNoFOV.txt', names=True, usecols=range(3, 8)),
    'acceptanceNoFOV': np.genfromtxt(DATA_PATH + '/xmax/acceptanceNoFOV.txt', names=True, usecols=range(3, 11)),
    'energyBins': np.r_[np.linspace(17.8, 19.5, 18), 20],
    'energyCens': np.r_[np.linspace(17.85, 19.45, 17), 19.7],
    'xmaxBins': np.linspace(0, 2000, 101),
    'xmaxCens': np.linspace(10, 1990, 100)}

# Values for <Xmax>, sigma(Xmax) parameterization from [3,4,5]
# DXMAX_PARAMS[model] = (X0, D, xi, delta, p0, p1, p2, a0, a1, b)
DXMAX_PARAMS = {
    # from [3], pre-LHC
    'QGSJet01': (774.2, 49.7, -0.30, 1.92, 3852, -274, 169, -0.451, -0.0020, 0.057),
    # from [3], pre-LHC
    'QGSJetII': (781.8, 45.8, -1.13, 1.71, 3163, -237, 60, -0.386, -0.0006, 0.043),
    # from [3], pre-LHC
    'EPOS1.99': (809.7, 62.2, 0.78, 0.08, 3279, -47, 228, -0.461, -0.0041, 0.059),
    # from [3]
    'Sibyll2.1': (795.1, 57.7, -0.04, -0.04, 2785, -364, 152, -0.368, -0.0049, 0.039),
    # from [5], fit range lgE = 17 - 20
    'Sibyll2.1*': (795.1, 57.9, 0.06, 0.08, 2792, -394, 101, -0.360, -0.0019, 0.037),
    # from [4]
    'EPOS-LHC': (806.1, 55.6, 0.15, 0.83, 3284, -260, 132, -0.462, -0.0008, 0.059),
    # from [5], fit range lgE = 17 - 20
    'EPOS-LHC*': (806.1, 56.3, 0.47, 1.15, 3270, -261, 149, -0.459, -0.0005, 0.058),
    # from [4]
    'QGSJetII-04': (790.4, 54.4, -0.31, 0.24, 3738, -375, -21, -0.397, 0.0008, 0.046),
    # from [5], fit range lgE = 17 - 20
    'QGSJetII-04*': (790.4, 54.4, -0.33, 0.69, 3702, -369, 83, -0.396, 0.0010, 0.045)}

# ln(A) moments from [6]
# noinspection PyTypeChecker
DLNA = {
    'EPOS-LHC': np.genfromtxt(DATA_PATH + '/lnA/lnA_EPOS-LHC.txt', names=True),
    'QGSJetII-04': np.genfromtxt(DATA_PATH + '/lnA/lnA_QGSJetII-04.txt', names=True),
    'Sibyll2.1': np.genfromtxt(DATA_PATH + '/lnA/lnA_Sibyll2.1.txt', names=True)}

# mass groups from [7]
MASS_PROBABILITIES = {
    'EPOS-LHC': np.genfromtxt(DATA_PATH + '/comp/comp-eps-4-tot.dat', unpack=True),
    'QGSJetII-04': np.genfromtxt(DATA_PATH + '/comp/comp-q04-4-tot.dat', unpack=True),
    'Sibyll2.1': np.genfromtxt(DATA_PATH + '/comp/comp-s21-4-tot.dat', unpack=True)}


# ------------------  FUNCTIONS ----------------------
def gumbel_parameters(log10e, mass, model='EPOS-LHC'):
    """
    Location, scale and shape parameter of the Gumbel Xmax distribution from [1], equations 3.1 - 3.6.

    :param log10e: energy log10(E/eV)
    :type log10e: array_like
    :param mass: mass number
    :type mass: array_like
    :param model: hadronic interaction model
    :type model: string
    :return: mu (array_like, location paramater [g/cm^2]), sigma (array_like, scale parameter [g/cm^2]),
             lamda (array_like, shape parameter)
    :rtype: tuple
    """
    l_e = log10e - 19  # log10(E/10 EeV)
    ln_mass = np.log(mass)
    d = np.array([np.ones_like(mass), ln_mass, ln_mass ** 2])

    # Parameters for mu, sigma and lambda of the Gumble Xmax distribution from [1], table 1.
    #   'model' : {
    #       'mu'     : ((a0, a1, a2), (b0, b1, b2), (c0, c1, c2))
    #       'sigma'  : ((a0, a1, a2), (b0, b1, b2))
    #       'lambda' : ((a0, a1, a2), (b0, b1, b2))}
    params = {
        'QGSJetII': {
            'mu': ((758.444, -10.692, -1.253), (48.892, 0.02, 0.179), (-2.346, 0.348, -0.086)),
            'sigma': ((39.033, 7.452, -2.176), (4.390, -1.688, 0.170)),
            'lambda': ((0.857, 0.686, -0.040), (0.179, 0.076, -0.0130))},
        'QGSJetII-04': {
            'mu': ((761.383, -11.719, -1.372), (57.344, -1.731, 0.309), (-0.355, 0.273, -0.137)),
            'sigma': ((35.221, 12.335, -2.889), (0.307, -1.147, 0.271)),
            'lambda': ((0.673, 0.694, -0.007), (0.060, -0.019, 0.017))},
        'Sibyll2.1': {
            'mu': ((770.104, -15.873, -0.960), (58.668, -0.124, -0.023), (-1.423, 0.977, -0.191)),
            'sigma': ((31.717, 1.335, -0.601), (-1.912, 0.007, 0.086)),
            'lambda': ((0.683, 0.278, 0.012), (0.008, 0.051, 0.003))},
        'EPOS1.99': {
            'mu': ((780.013, -11.488, -1.906), (61.911, -0.098, 0.038), (-0.405, 0.163, -0.095)),
            'sigma': ((28.853, 8.104, -1.924), (-0.083, -0.961, 0.215)),
            'lambda': ((0.538, 0.524, 0.047), (0.009, 0.023, 0.010))},
        'EPOS-LHC': {
            'mu': ((775.589, -7.047, -2.427), (57.589, -0.743, 0.214), (-0.820, -0.169, -0.027)),
            'sigma': ((29.403, 13.553, -3.154), (0.096, -0.961, 0.150)),
            'lambda': ((0.563, 0.711, 0.058), (0.039, 0.067, -0.004))}}
    par = params[model]

    p0, p1, p2 = np.dot(par['mu'], d)
    mu = p0 + p1 * l_e + p2 * l_e ** 2
    p0, p1 = np.dot(par['sigma'], d)
    sigma = p0 + p1 * l_e
    p0, p1 = np.dot(par['lambda'], d)
    lambd = p0 + p1 * l_e

    return mu, sigma, lambd


def gumbel(x, log10e, mass, model='EPOS-LHC', scale=(1, 1, 1)):
    """
    Gumbel Xmax distribution from [1], equation 2.3.

    :param x: Xmax in [g/cm^2]
    :param log10e: energy log10(E/eV)
    :param mass: mass number
    :param model: hadronic interaction model
    :param scale: scale parameters (mu, sigma, lambda) to evaluate
                  the impact of systematical uncertainties
    :return: G(xmax) : value of the Gumbel distribution at xmax.
    """
    mu, sigma, lambd = gumbel_parameters(log10e, mass, model)

    # scale parameters
    mu *= scale[0]
    sigma *= scale[1]
    lambd *= scale[2]

    z = (x - mu) / sigma
    return 1. / sigma * lambd ** lambd / scipy.special.gamma(lambd) * np.exp(-lambd * (z + np.exp(-z)))


def gumbel_cdf(x, log10e, mass, model='EPOS-LHC', scale=(1, 1, 1)):
    """
    Integrated Gumbel Xmax distribution from [2]

    :param x: upper limit Xmax in [g/cm^2]
    :param log10e: energy log10(E/eV)
    :param mass: mass number
    :param model: hadronic interaction model
    :param scale: scale parameters (mu, sigma, lambda) to evaluate
                  the impact of systematical uncertainties
    :return: integral -inf, x of G(xmax) : value of the Gumbel distribution
    """
    mu, sigma, lambd = gumbel_parameters(log10e, mass, model)

    # scale paramaters
    mu *= scale[0]
    sigma *= scale[1]
    lambd *= scale[2]

    z = (x - mu) / sigma
    return scipy.special.gammaincc(lambd, lambd * np.exp(-z))


def gumbel_sf(x, log10e, mass, model='EPOS-LHC', scale=(1, 1, 1)):
    """
    Integrated Gumbel Xmax distribution from [2]

    :param x: lower limit Xmax in [g/cm^2]
    :param log10e: energy log10(E/eV)
    :param mass: mass number
    :param model: hadronic interaction model
    :param scale: scale parameters (mu, sigma, lambda) to evaluate
                  the impact of systematical uncertainties
    :return: integral x, inf of G(xmax) : value of the Gumbel distribution
    """
    mu, sigma, lambd = gumbel_parameters(log10e, mass, model)

    # scale parameters
    mu *= scale[0]
    sigma *= scale[1]
    lambd *= scale[2]

    z = (x - mu) / sigma
    return scipy.special.gammainc(lambd, lambd * np.exp(-z))


def rand_xmax(log10e, mass, size=None, model='EPOS-LHC'):
    """
    Random Xmax values for given energy E [EeV] and mass number A, cf. [1].

    :type log10e : array_like
    :param log10e : energy log10(E/eV)
    :type mass : array_like
    :param mass : mass number
    :type model: string
    :param model: hadronic interaction model
    :type size: int or None
    :param size: number of xmax values to create
    :return xmax : random Xmax values in [g/cm^2]
    :rtype: array_like
    """
    mu, sigma, lambd = gumbel_parameters(log10e, mass, model)

    # From [2], theorem 3.1:
    # Y = -ln X is generalized Gumbel distributed for Erlang distributed X
    # Erlang is a special case of the gamma distribution
    return mu - sigma * np.log(np.random.gamma(lambd, 1. / lambd, size=size))


def rand_gumbel(log10e, mass, size=None, model='EPOS-LHC'):  # pragma: no cover
    """ Deprecated funcion -> See rand_xmax() """
    print("User warning: function rand_gumbel() is deprecated. Please use rand_xmax() in future!")
    return rand_xmax(log10e, mass, size=size, model=model)


def xmax_energy_bin(log10e):
    """Get xmax energy bin for given log10(energy) log10"""
    if (log10e < 17.8) or (log10e > 20):
        raise ValueError("Energy out of range log10(E/eV) = 17.8 - 20")
    return max(0, DXMAX['energyBins'].searchsorted(log10e) - 1)


def xmax_resolution(x, log10e, zsys=0, fov_cut=True):
    """
    Xmax resolution from [4] parametrized as a double Gaussian R(Xmax^rec - Xmax) = f*N(sigma1) + (1-f)*N(sigma2)

    :param x: Xmax,rec in [g/cm^2]
    :param log10e: log10(E/eV)
    :param zsys: standard score of systematical deviation
    :param fov_cut: was the fiducial volume cut applied?
    :return: Resolution pdf
    """
    i = xmax_energy_bin(log10e)
    key = 'resolution' if fov_cut else 'resolutionNoFOV'
    s1, s1err, s2, s2err, k = DXMAX[key][i]

    # uncertainties are correlated
    s1 += zsys * s1err
    s2 += zsys * s2err

    g1 = norm.pdf(x, 0, s1)
    g2 = norm.pdf(x, 0, s2)
    return k * g1 + (1 - k) * g2


def xmax_acceptance(x, log10e, zsys=0, fov_cut=True):
    """
    Xmax acceptance from [4] parametrized as a constant with exponential tails
                | exp(+ (Xmax - x1) / lambda1)       Xmax < x1
    eps(Xmax) = | 1                             for  x1 < Xmax < x2
                | exp(- (Xmax - x2) / lambda2)       Xmax > x2

    :param x: Xmax,true in [g/cm^2]
    :param log10e: log10(E/eV)
    :param zsys: standard score of systematical deviation
    :param fov_cut: was the fiducial volume cut applied?
    :return: Relative acceptance between 0 - 1
    """
    i = xmax_energy_bin(log10e)
    key = 'acceptance' if fov_cut else 'acceptanceNoFOV'
    x1, x1err, x2, x2err, l1, l1err, l2, l2err = DXMAX[key][i]

    # evaluating extreme cases, cf. xmax2014/README
    x1 -= zsys * x1err
    x2 += zsys * x2err
    l1 += zsys * l1err
    l2 += zsys * l2err

    x = np.array(x, dtype=float)
    lo = x < x1
    hi = x > x2
    acceptance = np.ones_like(x)
    acceptance[lo] = np.exp(+(x[lo] - x1) / l1)
    acceptance[hi] = np.exp(-(x[hi] - x2) / l2)
    return acceptance


def xmax_scale(log10e, zsys):
    """
    Systematic uncertainty dX on the Xmax scale from [4] Xmax,true is estimated to be within [sigma-, sigma+] of
    the measured value.

    :param log10e: log10(E/eV)
    :param zsys: standard score of systematical deviation
    :return: Systematical deviation dX = Xmax,true - Xmax,measured
    """
    i = xmax_energy_bin(log10e)
    up, lo = DXMAX['systematics'][i]
    shift = up if (zsys > 0) else -lo
    return zsys * shift


def mean_xmax(log10e, mass, model='EPOS-LHC'):
    """
    <Xmax> values for given energies log10e(E / eV), mass numbers A
    and hadronic interaction model, according to [3,4].
    """
    x0, d, xi, delta = DXMAX_PARAMS[model][:4]
    l_e = log10e - 19
    return x0 + d * l_e + (xi - d / np.log(10) + delta * l_e) * np.log(mass)


def var_xmax(log10e, mass, model='EPOS-LHC'):
    """
    Shower to shower fluctuations sigma^2_sh(Xmax) values for given energies
    log10e(E / eV), mass numbers A and hadronic interaction model, according to [3,4].
    """
    p0, p1, p2, a0, a1, b = DXMAX_PARAMS[model][4:]
    l_e = log10e - 19
    ln_mass = np.log(mass)
    s2p = p0 + p1 * l_e + p2 * l_e ** 2
    a = a0 + a1 * l_e
    return s2p * (1 + a * ln_mass + b * ln_mass ** 2)


def ln_a_moments(log10e, mass, weights=None, bins=DXMAX['energyBins']):
    """
    Energy binned <lnA> and sigma^2(lnA) distribution

    :type log10e: array_like
    :param log10e: energies in [EeV]
    :type mass: array_like
    :param mass: mass numbers
    :type weights: array_like
    :param weights: optional weights
    :type bins: array_like
    :param bins: energies bins in log10(E/eV)
    :returns: tuple consisting of:

              - energy bin centers in log10(E/eV), array_like
              - <ln(A)>, mean of ln(A), array_like
              - sigma^2(ln(A)), variance of ln(A) including shower to shower fluctuations, array_like

    """
    l_ec = (bins[1:] + bins[:-1]) / 2  # bin centers in log10(E / eV)
    mln_a, vln_a = stat.binned_mean_and_variance(log10e, np.log(mass), bins, weights)
    return l_ec, mln_a, vln_a


def ln_a_moments2xmax_moments(log10e, m_ln_mass, v_ln_mass, model='EPOS-LHC'):
    """
    Translate <lnA> & Var(lnA) into <Xmax> & Var(Xmax) according to [3,4].


    :param log10e: Array of energy bin centers in log10(E/eV)
    :param m_ln_mass: <ln(A)> in the corresponding energy bins
    :param v_ln_mass: Var(ln(A)) in the corresponding energy bins
    :param model: Hadronic interaction model
    :return: Tuple consisting of

            - mXmax : <Xmax> in the corresponding energy bins
            - vXmax : Var(Xmax) in the corresponding energy bins
    """
    l_e_e0 = log10e - 19  # energy bin centers in log10(E / 10 EeV)
    x0, d, xi, delta, p0, p1, p2, a0, a1, b = DXMAX_PARAMS[model]

    f_e = (xi - d / np.log(10) + delta * l_e_e0)
    sigma2_p = p0 + p1 * l_e_e0 + p2 * (l_e_e0 ** 2)
    a = a0 + a1 * l_e_e0

    m_xmax = x0 + d * l_e_e0 + f_e * m_ln_mass
    v_xmax = sigma2_p * (1 + a * m_ln_mass + b * (v_ln_mass + m_ln_mass ** 2)) + f_e ** 2 * v_ln_mass
    return m_xmax, v_xmax


def xmax_moments(log10e, mass, weights=None, model='EPOS-LHC', bins=DXMAX['energyBins']):
    """
    Energy binned <Xmax>, sigma^2(Xmax), cf. arXiv:1301.6637

    :param log10e: Array of energies in [eV]
    :param mass: Array of mass numbers
    :param weights: Array of weights (optional)
    :param model: Hadronic interaction model
    :param bins: Array of energies in log10(E/eV) defining the bin boundaries
    :return: tuple consisting of

              - lEc : Array of energy bin centers in log10(E/eV)
              - mXmax : Array of <Xmax> in the energy bins of lEc
              - vXmax : Array of sigma^2(Xmax) in the energy bins of lEc
    """
    lec, mln_a, vln_a = ln_a_moments(log10e, mass, weights, bins)
    m_xmax, v_xmax = ln_a_moments2xmax_moments(lec, mln_a, vln_a, model)
    return lec, m_xmax, v_xmax


def xmax_moments2ln_a_moments(log10e, m_xmax, v_xmax, model='EPOS-LHC'):
    """
    Translate <Xmax> & Var(Xmax) into <lnA> & Var(lnA) according to [3,4].

    :param log10e: Array of energy bin centers in log10(E/eV)
    :param m_xmax: <Xmax> in the corresponding energy bins
    :param v_xmax: Var(Xmax) in the corresponding energy bins
    :param model: Hadronic interaction model
    :return: tuple consisting of

             - mlnA : <ln(A)> in the corresponding energy bins
             - vlnA : Var(ln(A)) in the corresponding energy bins
    """
    lg_e_e0 = log10e - 19  # energy bin centers in log10(E / 10 EeV)
    x0, d, xi, delta, p0, p1, p2, a0, a1, b = DXMAX_PARAMS[model]

    a = a0 + a1 * lg_e_e0
    f_e = xi - d / np.log(10) + delta * lg_e_e0
    sigma2_p = p0 + p1 * lg_e_e0 + p2 * lg_e_e0 ** 2

    m_xmax_p = x0 + d * lg_e_e0
    mln_a = (m_xmax - m_xmax_p) / f_e
    sigma2_sh = sigma2_p * (1 + a * mln_a + b * mln_a ** 2)
    vln_a = (v_xmax - sigma2_sh) / (b * sigma2_p + f_e ** 2)
    return mln_a, vln_a


def rand_charge_from_auger(log10e, model='EPOS-LHC', smoothed=None):
    """
    Samples random energy dependent charges from Auger's Xmax measurements (arXiv: 1409.5083).

    :param log10e: Input energies (in log10(E / eV)), array-like. Output charges have same size.
    :param model: Hadronic interaction model ['EPOS-LHC', 'QGSJetII-04', 'Sibyll2.1']
    :param smoothed: if True, smoothes the charge number (instead binned into [1, 2, 7, 26])
    :return: charges in same size as log10e
    """
    d = MASS_PROBABILITIES[model]
    idx = np.array([1, 6, 11, 16])
    log10e_bins = np.log10(d[0])
    fmax = d[idx + 0]

    # Charges of proton, helium, nitrogen, iron
    z = np.array([1, 2, 7, 26])
    charges = np.zeros(log10e.size)
    indices = np.argmin(np.abs(np.array([log10e]) - np.array([log10e_bins]).T), axis=0)

    for i, f in enumerate(fmax.T):
        mask = indices == i
        n = np.sum(mask)
        if n == 0:
            continue
        charges[mask] = np.random.choice(z, n, p=f / f.sum())

    # Smooth charges according to uniform distribution in ln(A)
    if smoothed is not None:
        charges[charges == 2] = np.random.choice([2, 3], np.sum(charges == 2))
        charges[charges == 7] = np.random.choice(4 + np.array(range(10)), np.sum(charges == 7))
        charges[charges == 26] = np.random.choice(14 + np.array(range(13)), np.sum(charges == 26))

    return charges


def charge_fit_from_auger(log10e):
    """
    Gives charge fractions of H/He/CNO/Fe for highest energies oriented on Auger data (see GAP2016_047).

    :param log10e: Input energies (in log10(E / eV)), array-like. Output charges have same size of first dim.
    :return: charges in size (4, shape(log10e)), last dimension contains fractions in order H/He/CNO/Fe
    """

    e = pow(10, log10e-18)

    phi_p = 400*pow(e, 0.3)*np.exp(-e/5.)
    phi_he = 6*pow(e, 2.5)*np.exp(-e/5.)
    phi_cno = 10*pow(e, 1.7)*np.exp(-e/15.)
    phi_fe = 0.5*pow(e, 1.7)*np.exp(-e/40.)

    fractions = np.array([phi_p, phi_he, phi_cno, phi_fe])
    return fractions / np.sum(fractions)


def spectrum(log10e, weights=None, bins=np.linspace(17.5, 20.5, 31), normalize2bin=None):
    """
    Differential spectrum for given energies [log10(E / eV)] and optional weights.
    Optionally normalize to Auger spectrum in given bin.
    """
    # noinspection PyTypeChecker
    n, bins = np.histogram(log10e, bins, weights=weights)
    bin_widths = 10 ** bins[1:] - 10 ** bins[:-1]  # linear bin widths
    flux = n / bin_widths  # make differential
    if normalize2bin:
        c = DSPECTRUM['mean'][normalize2bin] / flux[normalize2bin]
        flux *= c
    return flux


def spectrum_analytic(log10e):
    """
    returns a analytic parametrization of the Auger energy spectrum
    units are 1/(eV km^2 sr yr), input is the cosmic-ray energy in log10(E / eV)
    """
    p = DSPECTRUM_ANALYTIC  # type: np.ndarray
    # noinspection PyTypeChecker
    energy = 10 ** log10e  # type: np.ndarray
    return np.where(energy < p[1],
                    p[0] * (energy / p[1]) ** (-p[3]),
                    p[0] * (energy / p[1]) ** (-p[4]) * (1 + (p[1] / p[2]) ** p[5]) *
                    (1 + (energy / p[2]) ** p[5]) ** -1)


def rand_energy_from_auger(n, log10e_min=17.5, log10e_max=None, ebin=0.001):
    """
    Returns energies from the analytic parametrization of the Auger energy spectrum
    units are 1/(eV km^2 sr yr)

    :param n: size of the sample
    :param log10e_min: minimal log10(energy) of the sample
    :param log10e_max: maximal log10(energy) of the sample: e<emax
    :param ebin: binning of the sampled energies
    :return: array of energies (in log10(E / eV))
    """
    log10e_max = 20.5 if log10e_max is None else log10e_max
    if log10e_max < log10e_min:
        raise Exception("log10e_max smaller than log10e_min.")

    log10e_bins = np.arange(log10e_min, log10e_max + ebin, ebin)
    d_n = 10 ** log10e_bins * spectrum_analytic(log10e_bins)
    log10e = np.random.choice(log10e_bins, n, p=d_n / d_n.sum())

    return log10e


# --------------------- PLOTTING functions -------------------------
def plot_spectrum(ax=None, scale=3, with_scale_uncertainty=False):  # pragma: no cover
    """
    Plot the Auger spectrum.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    log10e = DSPECTRUM['logE']
    c = (10 ** log10e) ** scale
    flux = c * DSPECTRUM['mean']
    flux_high = c * DSPECTRUM['stathi']
    flux_low = c * DSPECTRUM['statlo']

    ax.errorbar(log10e[:-3], flux[:-3], yerr=[flux_low[:-3], flux_high[:-3]],
                fmt='ko', linewidth=1, markersize=8, capsize=0)
    ax.plot(log10e[-3:], flux_high[-3:], 'kv', markersize=8)  # upper limits

    ax.set_xlabel(r'$\log_{10}$($E$/eV)')
    yl = r'$J(E)$ [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{%g}$]' % (scale - 1)
    if scale != 0:
        yl = r'$E^{%g}\,$' % scale + yl
    ax.set_ylabel(yl)

    # marker for the energy scale uncertainty
    if with_scale_uncertainty:
        uncertainty = np.array((0.86, 1.14))
        x = 20.25 + np.log10(uncertainty)
        y = uncertainty ** scale * 1e38
        ax.plot(x, y, 'k', lw=0.8)
        ax.plot(20.25, 1e38, 'ko', ms=5)
        ax.text(20.25, 5e37, r'$\Delta E/E = 14\%$', ha='center', fontsize=12)


# Xmax moments
def plot_mean_xmax(ax=None, with_legend=True, models=None):     # pragma: no cover
    """
    Plot the Auger <Xmax> distribution.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    models = ['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04'] if models is None else models
    d = DXMAX['moments']
    log10e = d['meanLgEnergy']
    m_x = d['mean_xmax']
    e_stat = d['mean_xmaxSigmaStat']
    e_syslo = d['mean_xmaxSigmaSysLow']
    e_syshi = d['mean_xmaxSigmaSysUp']

    l1 = ax.errorbar(log10e, m_x, yerr=e_stat, fmt='ko', lw=1, ms=8, capsize=0)
    l2 = ax.errorbar(log10e, m_x, yerr=[-e_syslo, e_syshi],
                     fmt='', lw=0, mew=1.2, c='k', capsize=5)

    ax.set_xlim(17.5, 20)
    ax.set_ylim(640, 840)
    ax.set_xlabel(r'$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g/cm$^2$]')

    if with_legend:
        legend1 = ax.legend((l1, l2),
                            (r'data $\pm\sigma_\mathrm{stat}$',
                             r'$\pm\sigma_\mathrm{sys}$'),
                            loc='upper left', fontsize=16, markerscale=0.8,
                            handleheight=1.4, handlelength=0.8)

    if models:
        l_e = np.linspace(17.5, 20.5, 100)
        ls = ('-', '--', ':')
        for i, m in enumerate(models):
            m_x1 = mean_xmax(l_e, 1, model=m)  # proton
            m_x2 = mean_xmax(l_e, 56, model=m)  # iron
            ax.plot(l_e, m_x1, 'k', lw=1, ls=ls[i], label=m)  # for legend
            ax.plot(l_e, m_x1, 'r', lw=1, ls=ls[i])
            ax.plot(l_e, m_x2, 'b', lw=1, ls=ls[i])

        if with_legend:
            ax.legend(loc='lower right', fontsize=14)
            # noinspection PyUnboundLocalVariable
            ax.add_artist(legend1)


def plot_std_xmax(ax=None, with_legend=True, models=None):  # pragma: no cover
    """
    Plot the Auger sigma(Xmax) distribution.
    """
    models = ['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04'] if models is None else models
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if models:
        l_e = np.linspace(17.5, 20.5, 100)
        ls = ('-', '--', ':')
        for i, m in enumerate(models):
            v_x1 = var_xmax(l_e, 1, model=m)  # proton
            v_x2 = var_xmax(l_e, 56, model=m)  # iron
            ax.plot(l_e, v_x1 ** .5, 'k', lw=1, ls=ls[i], label=m)  # for legend
            ax.plot(l_e, v_x1 ** .5, 'r', lw=1, ls=ls[i])
            ax.plot(l_e, v_x2 ** .5, 'b', lw=1, ls=ls[i])

    d = DXMAX['moments']
    l0g10e = d['meanLgEnergy']
    s_x = d['sigmaXmax']
    e_stat = d['sigmaXmaxSigmaStat']
    e_syslo = d['sigmaXmaxSigmaSysLow']
    e_syshi = d['sigmaXmaxSigmaSysUp']

    l1 = ax.errorbar(l0g10e, s_x, yerr=e_stat, fmt='ko', lw=1, ms=8, capsize=0)
    l2 = ax.errorbar(l0g10e, s_x, yerr=[-e_syslo, e_syshi],
                     fmt='', lw=0, mew=1.2, c='k', capsize=5)

    ax.set_xlabel(r'$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\sigma(\rm{X_{max}})$ [g/cm$^2$]')
    ax.set_xlim(17.5, 20)
    ax.set_ylim(1, 79)

    if with_legend:
        legend1 = ax.legend((l1, l2),
                            (r'data $\pm\sigma_\mathrm{stat}$',
                             r'$\pm\sigma_\mathrm{sys}$'),
                            loc='upper left', fontsize=16, markerscale=0.8,
                            handleheight=1.4, handlelength=0.8)
        if models:
            ax.legend(loc='lower right', fontsize=14)
            ax.add_artist(legend1)


def plot_xmax(ax=None, i=0):    # pragma: no cover
    """Plot a xmax distribution for energy bin i"""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    data = DXMAX['histograms'][i]
    bins = DXMAX['xmaxBins']
    cens = DXMAX['xmaxCens']
    ax.hist(cens, weights=data, bins=bins, histtype='step', color='k', lw=1.5)

    ax.set_xlabel(r'$X_\mathrm{max}$ [g/cm$^2$]')
    ax.set_ylabel('N')

    ebins = DXMAX['energyBins']
    info = r'$\log_{10}(E) = %.1f - %.1f$' % (ebins[i], ebins[i + 1])
    ax.text(0.98, 0.97, info, transform=ax.transAxes, ha='right', va='top')


def plot_xmax_all():    # pragma: no cover
    """Plot all xmax distributions"""
    # noinspection PyTypeChecker
    _, axes = plt.subplots(6, 3, sharex=True, figsize=(12, 20))
    axes = axes.flatten()
    for i in range(18):
        ax = axes[i]
        plot_xmax(ax, i)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks((600, 800, 1000))
        # ax.locator_params(axis='y', nbins=5, min=0)

    axes[16].set_xlabel(r'$X_\mathrm{max}$ [g/cm$^2$]')
    axes[6].set_ylabel(r'events / (20 g/cm$^2$)')


def plot_mean_ln_a(ax=None, model='EPOS-LHC', with_legend=True, with_comparison=True):  # pragma: no cover
    """
    Plot the Auger <lnA> distribution.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    d = DLNA[model]
    log10e = d['logE']
    mln_a = d['mlnA']
    e_stat = d['mlnAstat']
    e_syslo = d['mlnAsyslo'] - mln_a
    e_syshi = d['mlnAsyshi'] - mln_a

    ax.errorbar(log10e, mln_a, yerr=e_stat, fmt='ko', lw=1.2, ms=8, mew='0',
                label=r'data $\pm\sigma_\mathrm{stat}$ (%s)' % model)
    ax.errorbar(log10e, mln_a, yerr=[-e_syslo, e_syshi], fmt='', lw=0,
                mew=1.2, c='k', capsize=5, label=r'$\pm\sigma_\mathrm{sys}$')

    ax.set_xlim(17.5, 20)
    ax.set_ylim(-0.5, 4.2)
    ax.set_xlabel(r'$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\langle \ln A \rangle$')

    if with_comparison:
        # noinspection PyUnresolvedReferences
        trans = plt.matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        ln_a = np.log(np.array([1, 4, 14, 56]))
        name = ['p', 'He', 'N', 'Fe']
        for i in range(4):
            ax.axhline(ln_a[i], c='k', ls=':')
            ax.text(0.98, ln_a[i] - 0.05, name[i],
                    transform=trans, va='top', ha='right', fontsize=14)

    if with_legend:
        legend1 = ax.legend(loc='upper left', fontsize=16, markerscale=0.8,
                            handleheight=1.4, handlelength=0.8, frameon=True)
        frame = legend1.get_frame()
        frame.set_edgecolor('white')


def plot_var_ln_a(ax=None, model='EPOS-LHC', with_legend=True):     # pragma: no cover
    """
    Plot the Auger Var(lnA) distribution.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    d = DLNA[model]
    log10e = d['logE']
    vln_a = d['vlnA']
    e_stat = d['vlnAstat']
    e_syslo = d['vlnAsyslo'] - vln_a
    e_syshi = d['vlnAsyshi'] - vln_a

    ax.errorbar(log10e, vln_a, yerr=e_stat, fmt='ko', lw=1.2, ms=8, mew='0',
                label=r'data $\pm\sigma_\mathrm{stat}$ (%s)' % model)
    ax.errorbar(log10e, vln_a, yerr=[-e_syslo, e_syshi], fmt='', lw=0,
                mew=1.2, c='k', capsize=5, label=r'$\pm\sigma_\mathrm{sys}$')

    ax.fill_between([17.5, 20.5], [-2, -2], hatch='/',
                    facecolor='white', edgecolor='grey')

    ax.set_xlim(17.5, 20)
    ax.set_ylim(-2, 4.2)
    ax.set_xlabel(r'$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$V(\ln A)$')

    if with_legend:
        legend1 = ax.legend(loc='upper left', fontsize=16, markerscale=0.8,
                            handleheight=1.4, handlelength=0.8, frameon=True)
        frame = legend1.get_frame()
        frame.set_edgecolor('white')


# super plots
def plot_spectrum_xmax(scale=3, models=None):   # pragma: no cover
    """
    Plot spectrum and Xmax moments together
    """
    models = ['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04'] if models is None else models
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 16))
    fig.subplots_adjust(hspace=0, wspace=0)
    ax1, ax2, ax3 = axes

    plot_spectrum(ax1, scale, True)
    plot_mean_xmax(ax2, True, models)
    plot_std_xmax(ax3, False, models)

    ax1.semilogy()
    ax1.set_xlim(17.5, 20.5)
    ax1.set_ylim(8e35, 2e38)

    # model description
    ax2.text(19.0, 825, 'proton', fontsize=16, rotation=22)
    ax2.text(20.2, 755, 'iron', fontsize=16, rotation=23)
    ax3.text(20.4, 59, 'proton', fontsize=16, ha='right')
    ax3.text(20.4, 12, 'iron', fontsize=16, ha='right')

    for ax in axes:
        ax.axvline(18.7, c='grey', lw=1)  # ankle
    return fig, axes


def plot_spectrum_ln_a(scale=3, model='EPOS-LHC'):  # pragma: no cover
    """
    Plot spectrum and ln(A) moments together
    """
    # noinspection PyTypeChecker
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 16))
    fig.subplots_adjust(hspace=0, wspace=0)
    ax1, ax2, ax3 = axes

    plot_spectrum(ax1, scale, True)
    plot_mean_ln_a(ax2, model)
    plot_var_ln_a(ax3, model)

    ax1.semilogy()
    ax1.set_xlim(17.5, 20.5)
    ax1.set_ylim(8e35, 2e38)

    for ax in axes:
        ax.axvline(18.7, c='grey', lw=1)  # ankle
    return fig, axes


def plot_auger_exposure(color='g', markersize=2):   # pragma: no cover
    """
    Plot the outline of the geometrical Auger exposure with a maximum zenith angle of 60 degrees.
    Use this function after plotting/initialising a mollweide-projected plot.
    """
    theta = np.ones(1800) * np.deg2rad(60)
    phi = np.arange(1800)
    theta, phi = hp.Rotator(coord='cg')(theta, phi)
    hp.projplot(theta, phi, coord='G', color=color,
                marker='.', markersize=markersize)
