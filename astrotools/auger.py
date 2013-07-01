from astrotools import stat
import numpy as np
from matplotlib.pyplot import figure
import os.path


# --------------------- DATA -------------------------
cdir = os.path.split(__file__)[0]
#dSpectrum = np.genfromtxt(os.path.join(cdir, 'auger_spectrum11.txt'), delimiter=',', names=True)
#dXmax = np.genfromtxt(os.path.join(cdir, 'auger_xmax11.txt'), delimiter=',', names=True)
dSpectrum = np.genfromtxt(os.path.join(cdir, 'auger_spectrum13.txt'), delimiter=',', names=True)
dXmax = np.genfromtxt(os.path.join(cdir, 'auger_xmax13.txt'), delimiter=',', names=True)
xmaxBins = np.r_[ dXmax['logElo'].copy(), dXmax['logEhi'][-1] ]

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

# Parameters for gumble - Xmax distribution, cf. ARXIV:1305.2331 table 1.
# gumbelParams[model]['mu'] = ((a0, a1, a2), (b0, b1, b2), (c0, c1, c2))
# gumbelParams[model]['sigma'] = ((a0, a1, a2), (b0, b1, b2))
gumbelParams = {
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
# --------------------- DATA -------------------------


def geometricExposure(declination):
    """
    Auger geometric exposure for a given equatorial declination (see astro-ph/0004016).
    geometricExposure(declination (pi/2,-pi)) -> (0-1)
    """
    declination = np.array(declination)
    if (abs(declination) > np.pi/2).any():
        raise Exception('geometricExposure: declination not in range (pi/2, -pi/2)')
    zmax = np.deg2rad(60.0)
    olat = np.deg2rad(-35.25)
    xi = (np.cos(zmax) - np.sin(olat) * np.sin(declination)) / (np.cos(olat) * np.cos(declination))
    xi = np.clip(xi, -1, 1)
    am = np.arccos(xi)
    exposure = np.cos(olat) * np.cos(declination) * np.sin(am) + am * np.sin(olat) * np.sin(declination)
    return exposure / 1.8131550872084088

def randDeclination(n=1):
    """
    Returns n random declinations (0,pi) drawn from the Auger exposure.
    """
    # estimate number of required trials, exposure is about 1/3 of the sky
    nTry = int(3.3 * n) + 50
    dec = np.arcsin( 2*np.random.rand(nTry) - 1 )
    accept = geometricExposure(dec) > np.random.rand(nTry)
    if sum(accept) < n:
        raise Exception("randDeclination: stochastic failure")
    return dec[accept][:n]

def randXmax(E, A, model='Epos-LHC'):
    """
    Random Xmax values for given energy E [EeV] and mass number A (cf. ARXIV:1305.2331).
    """
    lE = np.log10(E/10.) # log10(E / 10 EeV)
    lnA = np.log(A)
    D = np.array([np.ones(np.shape(A)), lnA, lnA**2])

    p = gumbelParams[model]

    a, b, c = p['mu']
    p0 = np.dot(a, D)
    p1 = np.dot(b, D)
    p2 = np.dot(c, D)
    mu = p0 + p1*lE + p2*lE**2

    a, b = p['sigma']
    p0 = np.dot(a,D)
    p1 = np.dot(b,D)
    sigma = p0 + p1*lE

    a, b = p['lambda']
    p0 = np.dot(a,D)
    p1 = np.dot(b,D)
    lambd = p0 + p1*lE

    # cf. Kragujevac J. Math. 25 (2003) 19-29, theorem 3.1:
    # Y = -ln X is generalized Gumbel distributed for Erlang distributed X
    # Erlang is a special case of the gamma distribution
    return mu - sigma * np.log( np.random.gamma(lambd, 1./lambd) )

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
    E : Array of energies in [EeV]
    A : Array of mass numbers
    weights : Array of weights (optional)
    bins: Array of energies in log10(E/eV) defining the bin boundaries

    Returns
    -------
    lEc : Array of energy bin centers in log10(E/eV)
    mlnA : Array of <ln(A)> in the energy bins of lEc
    vlnA : Array of sigma^2(ln(A)) in the energy bins of lEc
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

def spectrum(E, weights=None, normalize2bin=None):
    """
    Differential spectrum for given energies [EeV] and optional weights.
    Optionally normalize to Auger spectrum in given bin.
    """
    lEbins = np.linspace(18.0, 20.5, 26)
    N, bins = np.histogram(np.log10(E) +18, lEbins, weights=weights)
    binWidths = 10**bins[1:] - 10**bins[:-1] # linear bin widths
    J = N / binWidths # make differential
    if normalize2bin:
        c = dSpectrum['mean'][normalize2bin] / J[normalize2bin]
        J *= c
    return J

def spectrumGroups(E, A, weights=None, normalize2bin=None):
    # indentify mass groups
    idx1 = A == 1
    idx2 = (A >= 2) * (A <= 8)
    idx3 = (A >= 9) * (A <= 26)
    idx4 = (A >= 27)

    # spectrum (non-differential) with same bins as the Auger spectrum
    lE = np.log10(E) + 18
    lEbins = np.linspace(18, 20.5, 26)
    N = np.histogram(lE, weights=weights, bins=lEbins)[0]

    if weights == None:
      N1 = np.histogram(lE[idx1], bins=lEbins)[0]
      N2 = np.histogram(lE[idx2], bins=lEbins)[0]
      N3 = np.histogram(lE[idx3], bins=lEbins)[0]
      N4 = np.histogram(lE[idx4], bins=lEbins)[0]
    else:
      N1 = np.histogram(lE[idx1], weights=weights[idx1], bins=lEbins)[0]
      N2 = np.histogram(lE[idx2], weights=weights[idx2], bins=lEbins)[0]
      N3 = np.histogram(lE[idx3], weights=weights[idx3], bins=lEbins)[0]
      N4 = np.histogram(lE[idx4], weights=weights[idx4], bins=lEbins)[0]

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
def plotSpectrum():
    """
    Plot of the Auger spectrum * E^3
    """
    logE = dSpectrum['logE']
    c = (10**logE)**3 # scale with E^3
    J = c * dSpectrum['mean']
    Jhi = c * dSpectrum['stathi']
    Jlo = c * dSpectrum['statlo']

    args = {'linewidth':1, 'markersize':8, 'markeredgewidth':0,}
    fig = figure()
    ax = fig.add_subplot(111)
    ax.errorbar(logE, J, yerr=[Jlo, Jhi], fmt='ko', **args)
#    ax.errorbar(logE[:22], J[:22], yerr=[Jlo[:22], Jhi[:22]], fmt='ko', **args)
#    ax.plot(logE[22:], Jhi[22:], 'kv', **args) # upper limits
    ax.set_xlabel('$\log_{10}$(E/[eV])')
    ax.set_ylabel('E$^3$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^2$]')
    ax.set_ylim((1e36, 1e38))
    ax.semilogy()
    return fig

def plotMeanXmax():
    """
    Plot the Auger <Xmax> distribution along with all distributions in xList.
    """
    fig = figure()
    ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(dXmax['logE'], dXmax['mean'], yerr=dXmax['mstat'], fmt='ko', **kwargs)
    lo = dXmax['mean']-dXmax['msyslo']
    hi = dXmax['mean']+dXmax['msyshi']
    ax.fill_between(dXmax['logE'], lo, hi, color='k', alpha=0.1)
    ax.set_xlim(18, 20)
    ax.set_ylim(680, 830)
    ax.set_xlabel('$\log_{10}$(E/[eV])')
    ax.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g cm$^{-2}$]')
    return fig

def plotStdXmax():
    """
    Plot the Auger sigma(Xmax) distribution along with all distributions in xList.
    """
    fig = figure()
    ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(dXmax['logE'], dXmax['std'], yerr=dXmax['sstat'], fmt='ko', **kwargs)
    lo = dXmax['std']-dXmax['ssyslo']
    hi = dXmax['std']+dXmax['ssyshi']
    ax.fill_between(dXmax['logE'], lo, hi, color='k', alpha=0.1)
    ax.set_xlim(18, 20)
    ax.set_ylim(0, 70)
    ax.set_xlabel('$\log_{10}$(E/[eV])')
    ax.set_ylabel(r'$\sigma(\rm{X_{max}})$ [g cm$^{-2}$]')
    return fig
