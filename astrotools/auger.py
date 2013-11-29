import numpy as np
import stat, coord
from os import path
from matplotlib.pyplot import figure, gca
import scipy.special


# References
# [1] Manlio De Domenico et al., JCAP07(2013)050, doi:10.1088/1475-7516/2013/07/050
# [2] S. Adeyemi and M.O. Ojo, Kragujevac J. Math. 25 (2003) 19-29
# [3] Pierre Auger Collaboration, Phys. Rev. Lett. 104, 091101, doi:10.1103/PhysRevLett.104.091101
# [4] Long Xmax paper

# --------------------- DATA -------------------------
cdir = path.split(__file__)[0]
dSpectrum = np.genfromtxt(path.join(cdir, 'auger_spectrum13.txt'), delimiter=',', names=True)
dXmax = np.genfromtxt(path.join(cdir, 'auger_xmax13.txt'), delimiter=',', names=True)
xmaxBins = np.r_[ dXmax['logElo'].copy(), dXmax['logEhi'][-1] ]

# ------------------  FUNCTIONS ----------------------
def randDec(n=1):
    """
    Returns n random equatorial declinations (pi/2, -pi/2) drawn from the Auger exposure.
    See coord.exposureEquatorial
    """
    # sample probability distribution using the rejection technique
    nTry = int(3.3 * n) + 50
    dec = np.arcsin( 2*np.random.rand(nTry) - 1 )
    maxVal = 0.58
    accept = coord.exposureEquatorial(dec, a0, zmax) > np.random.rand(nTry) * maxVal
    if sum(accept) < n:
        raise Exception("randEqDec: stochastic failure")
    return dec[accept][:n]

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

def gumbel(xmax, lgE, A, model='Epos-LHC'):
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

    Returns
    -------
    G(xmax) : array_like
        value of the Gumbel distribution at xmax.
    """
    mu, sigma, lambd = gumbelParameters(lgE, A, model)
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

def xmaxResolution(xmax, lgE, syst=0):
    """
    Parameterization of Xmax resolution as double Gaussian, cf. [3], [4].
    From http://www-ik.fzk.de/~munger/Xmax/XmaxResolution
    
    Parameters
    ----------
    xmax : array_like
        Xmax values in [g/cm^2]
    lgE : float
        energy log10(E/eV)
    syst : 0, 1, -1
        systematics

    Returns
    -------
    reso : array_like
        resolution pdf
    """
    def systematics(lo, hi, syst):
        if syst == 0:
            return (lo + hi) / 2.
        elif syst == 1:
            return hi
        elif syst == -1:
            return lo

    def molecSigma(lgE, syst):
        horizRMS = 2.
        fluctUp = np.sqrt(horizRMS**2 + (7.44 + (lgE-18) * (8.4 - 7.44) / 1.5)**2)
        fluctLo = np.sqrt(horizRMS**2 +  (5.6 + (lgE-18) * (6.9 - 5.6)  / 1.5)**2)
        return systematics(fluctLo, fluctUp, syst)

    def vaodSigma(lgE, syst):
        horizRMS = 7. + (lgE-18.)*(7.4-7.)/1.5
        fluctLo = horizRMS
        fluctUp = np.sqrt(horizRMS**2 + (6.8 + (lgE-18) * (11 - 6.8) / 1.5)**2)
        return systematics(fluctLo, fluctUp, syst)

    def detSigma(lgE, syst):
        resoPars = {
            'sibyll100' : [11.8852, 20.2620, 2.83435],
            'sibyll5600': [7.67575, 26.2451, 2.22486],
            'qgs100'    : [8.53650, 21.8872, 2.11341],
            'qgs5600'   : [6.57545, 25.5843, 2.03009]}
        lgEShifts = np.log10([1./1.22, 1., 1.22])
        minReso = 1000
        maxReso = 0
        for p in resoPars.values():
            reso = np.sqrt(p[0]**2 + p[1]**2 * (lgE + lgEShifts - 17)**-p[2])
            minReso = min(minReso, np.min(reso))
            maxReso = max(maxReso, np.max(reso))
        return systematics(minReso, maxReso, syst)

    # fraction
    f1 = -13.4332 + 1.41483 * lgE -0.0352555 * lgE**2
    f2 = 1 - f1
    
    # mean
    mean2 = -434.412 + 42.582 * lgE -1.03153 * lgE**2
    mean1 = (0 - f2 * mean2) / f1

    # shape factor
    factor = -57.812 + 5.71596 * lgE -0.133404 * lgE**2
    
    # variance
    var = detSigma(lgE, syst)**2
    atmVar = molecSigma(lgE, syst)**2 + vaodSigma(lgE, syst)**2
    
    if syst != 0:
        var += atmVar

    sigma1 = np.sqrt((var - f1 * f2 * (mean1 - mean2)**2) / (f1 + f2 * factor**2))
    sigma2 = sigma1 * factor

    if syst == 0:
        sigma1 = np.sqrt(sigma1**2 + atmVar)
        sigma2 = np.sqrt(sigma2**2 + atmVar)

    expo1 = f1 / sigma1 * np.exp(-((xmax - mean1) / sigma1)**2 / 2)
    expo2 = f2 / sigma2 * np.exp(-((xmax - mean2) / sigma2)**2 / 2)
    return (expo1 + expo2) / np.sqrt(2*np.pi)

def xmaxAcceptance(x, lgE):
    """
    ICRC13 Xmax acceptance from M. Unger
    See http://www-ik.fzk.de/~munger/Xmax/acceptance.cc

    Parameters
    ----------
    x : array_like
        Xmax in [g/cm^2]
    lgE : float
        energy log10(E/eV)

    Returns
    -------
    acceptance : array_like
        acceptance from 0 - 1
    """
    if lgE < 17.8 or lgE > 20:
        print "Energy out of range log10(E/eV) = 17.8 - 20"
        return None

    pars = [
        [542.886, 73.8676, 884.127, 95.9344],
        [584.098, 140.025, 885.229, 105.849],
        [610.279, 181.176, 895.729, 106.064],
        [584.82,  182.514, 900.577, 110.237],
        [565.086, 161.923, 900.979, 119.782],
        [604.367, 235.849, 879.14,  132.25 ],
        [577.335, 204.995, 878.603, 130.564],
        [570,     230.026, 901.462, 132.796],
        [646.766, 340.867, 892.405, 141.779],
        [589.163, 288.872, 900.562, 143.263],
        [612.944, 385.692, 917.641, 150.635],
        [550,     339.697, 926.059, 150.769],
        [550,     372.425, 915.054, 158.379],
        [564.829, 432.595, 904.962, 183.935],
        [418.076, 328.959, 915.61,  185.289],
        [539.277, 436.292, 941.815, 181.009],
        [414.343, 208.05,  915.652, 189.538],
        [447.257, 394.165, 912.463, 204.698]]

    bins = np.r_[np.linspace(17.8, 19.5, 18), 20.]
    iE = bins.searchsorted(lgE) - 1
    x1, w1, x2, w2 = pars[iE]

    x = np.array(x, dtype=float)
    lo = x < x1 # indices with Xmax < x1
    hi = x > x2 #              Xmax > x2
    acceptance = np.ones_like(x, )
    acceptance[lo] = np.exp( (x[lo] - x1) / w1)
    acceptance[hi] = np.exp(-(x[hi] - x2) / w2)
    return acceptance

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
def plotSpectrum(ax=None):
    """
    Plot the Auger spectrum.
    """
    if ax == None:
        fig = figure()
        ax = fig.add_subplot(111)
    logE = dSpectrum['logE']
    c = (10**logE)**3 # scale with E^3
    J = c * dSpectrum['mean']
    Jhi = c * dSpectrum['stathi']
    Jlo = c * dSpectrum['statlo']
    args = {'linewidth':1, 'markersize':8, 'markeredgewidth':0,}
    ax.errorbar(logE, J, yerr=[Jlo, Jhi], fmt='ko', **args)
#    ax.errorbar(logE[:22], J[:22], yerr=[Jlo[:22], Jhi[:22]], fmt='ko', **args)
#    ax.plot(logE[22:], Jhi[22:], 'kv', **args) # upper limits
    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel('E$^3$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^2$]')
    ax.set_ylim((1e36, 1e38))
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
    ax.fill_between(dXmax['logE'], lo, hi, color='k', alpha=0.1)
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
    ax.fill_between(dXmax['logE'], lo, hi, color='k', alpha=0.1)
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
