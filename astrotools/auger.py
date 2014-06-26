import numpy as np
from matplotlib.pyplot import figure, gca
from os import path
import scipy.special

import stat, coord

# References
# [1] Manlio De Domenico et al., JCAP07(2013)050, doi:10.1088/1475-7516/2013/07/050
# [2] S. Adeyemi and M.O. Ojo, Kragujevac J. Math. 25 (2003) 19-29
# [3] Pierre Auger Collaboration, Phys. Rev. Lett. 104, 091101, doi:10.1103/PhysRevLett.104.091101
# [4] Long Xmax paper

# --------------------- DATA -------------------------
cdir = path.split(__file__)[0]
dSpectrum = np.genfromtxt(path.join(cdir, 'auger_spectrum.txt'), delimiter=',', names=True)
dXmax = np.genfromtxt(path.join(cdir, 'auger_xmaxmoments.txt'), delimiter=',', names=True)
xmaxBins = np.r_[np.linspace(17.8, 19.5, 18), 19.9]

# ------------------  FUNCTIONS ----------------------
def gumbelParameters(lgE, A, model='Epos-LHC'):
    """
    Location, scale and shape parameter of the Gumbel Xmax distribution from [1], equations 3.1 - 3.6.

    Parameters
    ----------
    lgE : array_like
        energy log10(E/eV)
    A : array_like
        mass number
    model: string
        hadronic interaction model

    Returns
    -------
    mu : array_like
        location paramater [g/cm^2]
    sigma : array_like
        scale parameter [g/cm^2]
    lambda : array_like
        shape parameter
    """
    lE = lgE - 19 # log10(E/10 EeV)
    lnA = np.log(A)
    D = np.array([np.ones_like(A), lnA, lnA**2])

    # Parameters for mu, sigma and lambda of the Gumble Xmax distribution from [1], table 1.
    params = {
    #   'model' : {
    #       'mu'     : ((a0, a1, a2), (b0, b1, b2), (c0, c1, c2))
    #       'sigma'  : ((a0, a1, a2), (b0, b1, b2))
    #       'lambda' : ((a0, a1, a2), (b0, b1, b2))}
        'QGSJet II' : {
            'mu'     : ((758.444, -10.692, -1.253), (48.892, 0.02, 0.179), (-2.346, 0.348, -0.086)),
            'sigma'  : ((39.033, 7.452, -2.176), (4.390, -1.688, 0.170)),
            'lambda' : ((0.857, 0.686, -0.040), (0.179, 0.076, -0.0130))},
        'QGSJet II-04' : {
            'mu'     : ((761.383, -11.719, -1.372), (57.344, -1.731, 0.309), (-0.355, 0.273, -0.137)),
            'sigma'  : ((35.221, 12.335, -2.889), (0.307, -1.147, 0.271)),
            'lambda' : ((0.673, 0.694, -0.007), (0.060, -0.019, 0.017))},
        'Sibyll 2.1' : {
            'mu'     : ((770.104, -15.873, -0.960), (58.668, -0.124, -0.023), (-1.423, 0.977, -0.191)),
            'sigma'  : ((31.717, 1.335, -0.601), (-1.912, 0.007, 0.086)),
            'lambda' : ((0.683, 0.278, 0.012), (0.008, 0.051, 0.003))},
        'Epos 1.99' : {
            'mu'     : ((780.013, -11.488, -1.906), (61.911, -0.098, 0.038), (-0.405, 0.163, -0.095)),
            'sigma'  : ((28.853, 8.104, -1.924), (-0.083, -0.961, 0.215)),
            'lambda' : ((0.538, 0.524, 0.047), (0.009, 0.023, 0.010))},
        'Epos-LHC' : {
            'mu'     : ((775.589, -7.047, -2.427), (57.589, -0.743, 0.214), (-0.820, -0.169, -0.027)),
            'sigma'  : ((29.403, 13.553, -3.154), (0.096, -0.961, 0.150)),
            'lambda' : ((0.563, 0.711, 0.058), (0.039, 0.067, -0.004))}}
    par = params[model]

    p0, p1, p2 = np.dot(par['mu'], D)
    mu = p0 + p1*lE + p2*lE**2
    p0, p1 = np.dot(par['sigma'], D)
    sigma = p0 + p1*lE
    p0, p1 = np.dot(par['lambda'], D)
    lambd = p0 + p1*lE

    return mu, sigma, lambd

def gumbel(xmax, lgE, A, model='Epos-LHC', scale=(1,1,1)):
    """
    Gumbel Xmax distribution from [1], equation 2.3.

    Parameters
    ----------
    xmax : array_like
        Xmax in [g/cm^2]
    lgE : array_like
        energy log10(E/eV)
    A : array_like
        mass number
    model: string
        hadronic interaction model
    scale: array_like, 3 values
        scale parameters (mu, sigma, lambda) to evaluate
        the impact of systematical uncertainties

    Returns
    -------
    G(xmax) : array_like
        value of the Gumbel distribution at xmax.
    """
    mu, sigma, lambd = gumbelParameters(lgE, A, model)

    # scale paramaters
    mu    *= scale[0]
    sigma *= scale[1]
    lambd *= scale[2]

    z = (xmax - mu) / sigma
    return 1./sigma * lambd**lambd / scipy.special.gamma(lambd) * np.exp(-lambd * (z + np.exp(-z)))

def randGumbel(lgE, A, model='Epos-LHC'):
    """
    Random Xmax values for given energy E [EeV] and mass number A, cf. [1].

    Parameters
    ----------
    lgE : array_like
        energy log10(E/eV)
    A : array_like
        mass number
    model: string
        hadronic interaction model

    Returns
    -------
    xmax : array_like
        random Xmax values in [g/cm^2]
    """
    mu, sigma, lambd = gumbelParameters(lgE, A, model)
    # From [2], theorem 3.1:
    # Y = -ln X is generalized Gumbel distributed for Erlang distributed X
    # Erlang is a special case of the gamma distribution
    return mu - sigma * np.log( np.random.gamma(lambd, 1./lambd) )


def getEnergyBin(lgE):
    if lgE < 17.8 or lgE > 20:
        print "Energy out of range log10(E/eV) = 17.8 - 20"
        return None
    Ebins = np.r_[np.linspace(17.8, 19.5, 18), 20.]
    return Ebins.searchsorted(lgE) - 1

def xmaxResolution(x, lgE):
    """
    Xmax resolution from [1]
    Returns: resolution pdf
    """
    i = getEnergyBin(lgE)
    s1, es1, s2, es2, k = [
        [1.753e+01, 7.469e-01, 3.372e+01, 1.437e+00, 6.168e-01],
        [1.667e+01, 7.135e-01, 3.286e+01, 1.406e+00, 6.255e-01],
        [1.586e+01, 6.944e-01, 3.195e+01, 1.398e+00, 6.342e-01],
        [1.512e+01, 6.890e-01, 3.100e+01, 1.413e+00, 6.430e-01],
        [1.444e+01, 6.960e-01, 3.004e+01, 1.448e+00, 6.517e-01],
        [1.381e+01, 7.144e-01, 2.905e+01, 1.503e+00, 6.606e-01],
        [1.326e+01, 7.413e-01, 2.809e+01, 1.570e+00, 6.694e-01],
        [1.277e+01, 7.750e-01, 2.713e+01, 1.647e+00, 6.782e-01],
        [1.235e+01, 8.112e-01, 2.625e+01, 1.724e+00, 6.869e-01],
        [1.198e+01, 8.484e-01, 2.541e+01, 1.799e+00, 6.958e-01],
        [1.168e+01, 8.826e-01, 2.466e+01, 1.863e+00, 7.047e-01],
        [1.146e+01, 9.098e-01, 2.406e+01, 1.910e+00, 7.134e-01],
        [1.128e+01, 9.302e-01, 2.357e+01, 1.943e+00, 7.223e-01],
        [1.117e+01, 9.425e-01, 2.326e+01, 1.962e+00, 7.307e-01],
        [1.110e+01, 9.498e-01, 2.308e+01, 1.974e+00, 7.397e-01],
        [1.108e+01, 9.536e-01, 2.307e+01, 1.985e+00, 7.479e-01],
        [1.109e+01, 9.580e-01, 2.320e+01, 2.004e+00, 7.573e-01],
        [1.115e+01, 9.713e-01, 2.368e+01, 2.062e+00, 7.725e-01]][i]

    g1 = normpdf(x, 0, s1)
    g2 = normpdf(x, 0, s2)
    return k * g1 + (1-k) * g2

def xmaxAcceptance(x, lgE):
    """
    Xmax acceptance from [1]
    Returns: acceptance(x) between 0 - 1
    """
    i = getEnergyBin(lgE)
    x1, ex1, x2, ex2, l1, el1, l2, el2 = [
        [5.861e+02, 5.707e+00, 8.813e+02, 7.850e+00, 1.087e+02, 1.680e+01, 9.486e+01, 6.850e+00],
        [5.923e+02, 8.530e+00, 8.831e+02, 7.949e+00, 1.334e+02, 1.739e+01, 1.009e+02, 6.949e+00],
        [5.973e+02, 1.140e+01, 8.849e+02, 8.048e+00, 1.578e+02, 1.877e+01, 1.069e+02, 7.048e+00],
        [6.011e+02, 1.433e+01, 8.867e+02, 8.148e+00, 1.821e+02, 2.094e+01, 1.130e+02, 7.148e+00],
        [6.037e+02, 1.728e+01, 8.885e+02, 8.247e+00, 2.060e+02, 2.390e+01, 1.191e+02, 7.247e+00],
        [6.051e+02, 2.034e+01, 8.903e+02, 8.348e+00, 2.299e+02, 2.772e+01, 1.253e+02, 7.348e+00],
        [6.053e+02, 2.337e+01, 8.921e+02, 8.447e+00, 2.531e+02, 3.227e+01, 1.313e+02, 7.447e+00],
        [6.042e+02, 2.650e+01, 8.940e+02, 8.548e+00, 2.764e+02, 3.770e+01, 1.375e+02, 7.548e+00],
        [6.020e+02, 2.958e+01, 8.958e+02, 8.646e+00, 2.986e+02, 4.376e+01, 1.435e+02, 7.646e+00],
        [5.985e+02, 3.279e+01, 8.976e+02, 8.747e+00, 3.212e+02, 5.080e+01, 1.497e+02, 7.747e+00],
        [5.938e+02, 3.606e+01, 8.995e+02, 8.849e+00, 3.436e+02, 5.871e+01, 1.559e+02, 7.849e+00],
        [5.880e+02, 3.926e+01, 9.012e+02, 8.947e+00, 3.649e+02, 6.715e+01, 1.619e+02, 7.947e+00],
        [5.808e+02, 4.260e+01, 9.031e+02, 9.048e+00, 3.865e+02, 7.664e+01, 1.681e+02, 8.048e+00],
        [5.729e+02, 4.579e+01, 9.048e+02, 9.144e+00, 4.065e+02, 8.634e+01, 1.739e+02, 8.144e+00],
        [5.631e+02, 4.928e+01, 9.067e+02, 9.247e+00, 4.279e+02, 9.767e+01, 1.802e+02, 8.247e+00],
        [5.531e+02, 5.246e+01, 9.084e+02, 9.340e+00, 4.469e+02, 1.086e+02, 1.859e+02, 8.340e+00],
        [5.404e+02, 5.614e+01, 9.103e+02, 9.447e+00, 4.682e+02, 1.220e+02, 1.924e+02, 8.447e+00],
        [5.169e+02, 6.222e+01, 9.135e+02, 9.620e+00, 5.020e+02, 1.456e+02, 2.030e+02, 8.620e+00]][i]

    x = np.array(x, dtype=float)
    lo = x < x1 # indices with Xmax < x1
    hi = x > x2 #              Xmax > x2
    acceptance = np.ones_like(x, )
    acceptance[lo] = np.exp( (x[lo] - x1) / l1)
    acceptance[hi] = np.exp(-(x[hi] - x2) / l2)
    return acceptance

def xmaxSystematics(lgE):
    """
    Systematic uncertainty on Xmax
    Returns Xhi, Xlo
    """
    i = getEnergyBin(lgE)
    return [
        [7.488e+00, -1.014e+01],
        [7.324e+00, -1.010e+01],
        [7.158e+00, -1.004e+01],
        [6.997e+00, -9.927e+00],
        [6.851e+00, -9.770e+00],
        [6.723e+00, -9.554e+00],
        [6.623e+00, -9.291e+00],
        [6.547e+00, -8.978e+00],
        [6.498e+00, -8.647e+00],
        [6.470e+00, -8.296e+00],
        [6.460e+00, -7.955e+00],
        [6.464e+00, -7.653e+00],
        [6.478e+00, -7.386e+00],
        [6.498e+00, -7.182e+00],
        [6.524e+00, -7.013e+00],
        [6.551e+00, -6.903e+00],
        [6.583e+00, -6.821e+00],
        [6.639e+00, -6.759e+00]][i]



# Values for <Xmax>, sigma(Xmax) parameterization, cf. arXiv:1301.6637 tables 1 and 2.
# Parameters for Epos LHC and QGSJet II-04 are from the ICRC '13 Proceedings by Eun-Joo Ahn.
# xmaxParams[model] = (X0, D, xi, delta, p0, p1, p2, a0, a1, b)
xmaxParams = {
    'QGSJet 01'    : (774.2, 49.7, -0.30,  1.92, 3852, -274, 169, -0.451, -0.0020, 0.057),
    'QGSJet II'    : (781.8, 45.8, -1.13,  1.71, 3163, -237,  60, -0.386, -0.0006, 0.043),
    'QGSJet II-04' : (790.4, 54.4, -0.31,  0.24, 3738, -375, -21, -0.397,  0.0008, 0.046),
    'Sibyll 2.1'   : (795.1, 57.7, -0.04, -0.04, 2785, -364, 152, -0.368, -0.0049, 0.039),
    'Epos 1.99'    : (809.7, 62.2,  0.78,  0.08, 3279,  -47, 228, -0.461, -0.0041, 0.059),
    'Epos-LHC'     : (806.1, 55.6,  0.15,  0.83, 3284, -260, 132, -0.462, -0.0008, 0.059)}

def meanXmax(E, A, model='Epos-LHC'):
    """
    <Xmax> values for given energies E [EeV], mass numbers A and a hadronic interaction model.
    See arXiv:1301.6637
    """
    X0, D, xi, delta = xmaxParams[model][:4]
    lE = np.log10(E) - 1
    return X0 + D*lE + (xi - D/np.log(10) + delta*lE)*np.log(A)

def varXmax(E, A, model='Epos-LHC'):
    """
    Shower to shower fluctuations sigma^2_sh(Xmax) values for given energies E [EeV], mass numbers A and a hadronic interaction model.
    See arXiv:1301.6637
    """
    p0, p1, p2, a0, a1, b = xmaxParams[model][4:]
    lE = np.log10(E) - 1
    lnA = np.log(A)
    s2p = p0 + p1*lE + p2*(lE**2)
    a = a0 + a1*lE
    return s2p*( 1 + a*lnA + b*(lnA**2) )

def lnADistribution(E, A, weights=None, bins=xmaxBins):
    """
    Energy binned <lnA> and sigma^2(lnA) distribution

    Parameters
    ----------
    E : array_like
        energies in [EeV]
    A : array_like
        mass numbers
    weights : array_like, optional
        weights
    bins: array_like
        energies bins in log10(E/eV)

    Returns
    -------
    lEc : array_like
        energy bin centers in log10(E/eV)
    mlnA : array_like
        <ln(A)>, mean of ln(A)
    vlnA : array_like
        sigma^2(ln(A)), variance of ln(A) including shower to shower fluctuations
    """
    lE = np.log10(E) + 18 # event energies in log10(E / eV)
    lEc = (bins[1:] + bins[:-1]) / 2 # bin centers in log10(E / eV)
    mlnA, vlnA = stat.binnedMeanAndVariance(lE, np.log(A), bins, weights)
    return (lEc, mlnA, vlnA)

def lnA2XmaxDistribution(lEc, mlnA, vlnA, model='Epos-LHC'):
    """
    Convert an energy binned <lnA>, sigma^2(lnA) distribution to <Xmax>, sigma^2(Xmax), cf. arXiv:1301.6637

    Parameters
    ----------
    lEc : Array of energy bin centers in log10(E/eV)
    mlnA : Array of <ln(A)> in the energy bins of lEc
    vlnA : Array of sigma^2(ln(A)) in the energy bins of lEc
    model : Hadronic interaction model

    Returns
    -------
    mXmax : Array of <Xmax> in the energy bins of lEc
    vXmax : Array of sigma^2(Xmax) in the energy bins of lEc
    """
    lEc = lEc - 19 # energy bin centers in log10(E / 10 EeV)
    X0, D, xi, delta, p0, p1, p2, a0, a1, b = xmaxParams[model]

    fE = (xi - D/np.log(10) + delta*lEc)
    s2p = p0 + p1*lEc + p2*(lEc**2)
    a = a0 + a1*lEc

    mXmax = X0 + D*lEc + fE*mlnA # eq. 2.6
    vXmax = s2p*( 1 + a*mlnA + b*(vlnA + mlnA**2) ) + fE**2*vlnA # eq. 2.12
    return (mXmax, vXmax)

def xmaxDistribution(E, A, weights=None, model='Epos-LHC', bins=xmaxBins):
    """
    Energy binned <Xmax>, sigma^2(Xmax), cf. arXiv:1301.6637

    Parameters
    ----------
    E : Array of energies in [EeV]
    A : Array of mass numbers
    weights : Array of weights (optional)
    model : Hadronic interaction model
    bins: Array of energies in log10(E/eV) defining the bin boundaries

    Returns
    -------
    lEc : Array of energy bin centers in log10(E/eV)
    mXmax : Array of <Xmax> in the energy bins of lEc
    vXmax : Array of sigma^2(Xmax) in the energy bins of lEc
    """
    lEc, mlnA, vlnA = lnADistribution(E, A, weights, bins)
    mXmax, vXmax = lnA2XmaxDistribution(lEc, mlnA, vlnA, model)
    return (lEc, mXmax, vXmax)

def spectrum(E, weights=None, bins=np.linspace(17.5, 20.2, 28), normalize2bin=None):
    """
    Differential spectrum for given energies [EeV] and optional weights.
    Optionally normalize to Auger spectrum in given bin.
    """
    N, bins = np.histogram(np.log10(E) +18, bins, weights=weights)
    binWidths = 10**bins[1:] - 10**bins[:-1] # linear bin widths
    J = N / binWidths # make differential
    if normalize2bin:
        c = dSpectrum['mean'][normalize2bin] / J[normalize2bin]
        J *= c
    return J

def spectrumGroups(E, A, weights=None, bins=np.linspace(17.5, 20.2, 28), normalize2bin=None):
    # indentify mass groups
    idx1 = A == 1
    idx2 = (A >= 2) * (A <= 8)
    idx3 = (A >= 9) * (A <= 26)
    idx4 = (A >= 27)

    # spectrum (non-differential) with same bins as the Auger spectrum
    lE = np.log10(E) + 18
    N = np.histogram(lE, weights=weights, bins=bins)[0]

    if weights == None:
      N1 = np.histogram(lE[idx1], bins=bins)[0]
      N2 = np.histogram(lE[idx2], bins=bins)[0]
      N3 = np.histogram(lE[idx3], bins=bins)[0]
      N4 = np.histogram(lE[idx4], bins=bins)[0]
    else:
      N1 = np.histogram(lE[idx1], weights=weights[idx1], bins=bins)[0]
      N2 = np.histogram(lE[idx2], weights=weights[idx2], bins=bins)[0]
      N3 = np.histogram(lE[idx3], weights=weights[idx3], bins=bins)[0]
      N4 = np.histogram(lE[idx4], weights=weights[idx4], bins=bins)[0]

    # make spectrum differential and optionally scale to bin
    binwidths = binWidths = 10**lEbins[1:] - 10**lEbins[:-1]
    J = N / binWidths
    c = 1
    if normalize2bin:
        c = dSpectrum['mean'][normalize2bin] / J[normalize2bin]
        J *= c
    J1 = N1 / binWidths * c
    J2 = N2 / binWidths * c
    J3 = N3 / binWidths * c
    J4 = N4 / binWidths * c

    return [J, J1, J2, J3, J4]

# --------------------- PLOT -------------------------
def plotSpectrum(ax=None, scale=3):
    """
    Plot the Auger spectrum.
    """
    if ax == None:
        fig = figure()
        ax = fig.add_subplot(111)
    logE = dSpectrum['logE']
    c = (10**logE)**scale
    J = c * dSpectrum['mean']
    Jhi = c * dSpectrum['stathi']
    Jlo = c * dSpectrum['statlo']
    args = {'linewidth':1, 'markersize':8, 'markeredgewidth':0,}
    ax.errorbar(logE, J, yerr=[Jlo, Jhi], fmt='ko', **args)
    ax.errorbar(logE[:27], J[:27], yerr=[Jlo[:27], Jhi[:27]], fmt='ko', **args)
    ax.plot(logE[27:], Jhi[27:], 'kv', **args) # upper limits
    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel('E$^{%.1f}$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{%.1f}$]'%(scale, scale-1))
    ax.semilogy()

def plotMeanXmax(ax=None):
    """
    Plot the Auger <Xmax> distribution.
    """
    if ax == None:
        fig = figure()
        ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(dXmax['logE'], dXmax['mean'], yerr=dXmax['mstat'], fmt='ko', **kwargs)
    lo = dXmax['mean']-dXmax['msyslo']
    hi = dXmax['mean']+dXmax['msyshi']
    # ax.fill_between(dXmax['logE'], lo, hi, color='k', alpha=0.1)
    ax.set_xlim(17.8, 20)
    ax.set_ylim(680, 830)
    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g cm$^{-2}$]')

def plotStdXmax(ax=None):
    """
    Plot the Auger sigma(Xmax) distribution.
    """
    if ax == None:
        fig = figure()
        ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(dXmax['logE'], dXmax['std'], yerr=dXmax['sstat'], fmt='ko', **kwargs)
    lo = dXmax['std']-dXmax['ssyslo']
    hi = dXmax['std']+dXmax['ssyshi']
    # ax.fill_between(dXmax['logE'], lo, hi, color='k', alpha=0.1)
    ax.set_xlim(17.8, 20)
    ax.set_ylim(0, 80)
    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\sigma(\rm{X_{max}})$ [g cm$^{-2}$]')

def plotMeanXmaxModels(ax=None, models=('Epos-LHC', 'QGSJet II-04', 'Sibyll 2.1')):
    """
    Add expectations from simulations to the mean(Xmax) plot.
    If not given an axes object, it will take the current axes:
    """
    if ax == None:
        ax = gca()
    E = np.logspace(17.5, 20.5, 100) / 1e18
    A = np.ones(100)
    bins = np.linspace(17.5, 20.5, 50)
    ls = ('-', '--', '-.', '-', '--', '-.')
    for i, m in enumerate(models):
        cmp1 = xmaxDistribution(E,    A, bins=bins, model=m)
        cmp2 = xmaxDistribution(E, 56*A, bins=bins, model=m)
        ax.plot(cmp1[0], cmp1[1], 'b', lw=1, ls=ls[i])
        ax.plot(cmp2[0], cmp2[1], 'r', lw=1, ls=ls[i])

def plotStdXmaxModels(ax=None, models=('Epos-LHC', 'QGSJet II-04', 'Sibyll 2.1')):
    """
    Add expectations from simulations to the sigma(Xmax) plot.
    If not given an axes object, it will take the current axes.
    """
    if ax == None:
        ax = gca()
    E = np.logspace(17.5, 20.5, 100) / 1e18
    A = np.ones(100)
    bins = np.linspace(17.5, 20.5, 50)
    ls = ('-', '--', '-.', '-', '--', '-.')
    for i, m in enumerate(models):
        cmp1 = xmaxDistribution(E,    A, bins=bins, model=m)
        cmp2 = xmaxDistribution(E, 56*A, bins=bins, model=m)
        ax.plot(cmp1[0], cmp1[2]**.5, 'b', lw=1, ls=ls[i])
        ax.plot(cmp2[0], cmp2[2]**.5, 'r', lw=1, ls=ls[i])
