import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import normpdf
import scipy.special
import stat

# References
# [1] Manlio De Domenico et al., JCAP07(2013)050, doi:10.1088/1475-7516/2013/07/050
# [2] S. Adeyemi and M.O. Ojo, Kragujevac J. Math. 25 (2003) 19-29
# [3] Pierre Auger Collaboration, Phys. Rev. Lett. 104, 091101, doi:10.1103/PhysRevLett.104.091101
# [4] Long Xmax paper
# [5] Auger ICRC'13

# --------------------- DATA -------------------------
cdir = path.split(__file__)[0]

# Spectrum data [5]
dSpectrum = np.genfromtxt(cdir+'/auger_spectrum_2013.txt'), delimiter=',', names=True)

# Xmax data of [4], from http://www.auger.org/data/xmax2014.tar.gz on 2014-09-29
dXmax = {}
dXmax['histograms']  = np.genfromtxt(cdir+'/xmax2014/xmaxHistograms.txt', usecols=range(25,61))  # 500 < Xmax/[g/cm^2] < 1200
dXmax['moments']     = np.genfromtxt(cdir+'/xmax2014/xmaxMoments.txt', names=True, usecols=range(3,13))
dXmax['resolution']  = np.genfromtxt(cdir+'/xmax2014/resolution.txt', names=True, usecols=range(3,8))
dXmax['acceptance']  = np.genfromtxt(cdir+'/xmax2014/acceptance.txt', names=True, usecols=range(3,11))
dXmax['systematics'] = np.genfromtxt(cdir+'/xmax2014/xmaxSystematics.txt', names=True, usecols=(3,4))
# dXmax['correlationsPlus'] = ...
# dXmax['correlationsMinus'] = ...
dXmax['energyBins']  = np.r_[np.linspace(17.8, 19.5, 18), 19.9]
dXmax['energyCens']  = np.r_[np.linspace(17.85, 19.45, 17), 19.7]
dXmax['xmaxBins']    = np.linspace(0, 2000, 101)[25:61]
dXmax['xmaxCens']    = np.linspace(10, 1990, 100)[25:60]

# Values for <Xmax>, sigma(Xmax) parameterization, cf. arXiv:1301.6637 tables 1 and 2.
# Parameters for EPOS LHC and QGSJet II-04 are from the ICRC '13 Proceedings by Eun-Joo Ahn.
# xmaxParams[model] = (X0, D, xi, delta, p0, p1, p2, a0, a1, b)
xmaxParams = {
    'QGSJet01'    : (774.2, 49.7, -0.30,  1.92, 3852, -274, 169, -0.451, -0.0020, 0.057),
    'QGSJetII'    : (781.8, 45.8, -1.13,  1.71, 3163, -237,  60, -0.386, -0.0006, 0.043),
    'QGSJetII-04' : (790.4, 54.4, -0.31,  0.24, 3738, -375, -21, -0.397,  0.0008, 0.046),
    'Sibyll2.1'   : (795.1, 57.7, -0.04, -0.04, 2785, -364, 152, -0.368, -0.0049, 0.039),
    'EPOS1.99'    : (809.7, 62.2,  0.78,  0.08, 3279,  -47, 228, -0.461, -0.0041, 0.059),
    'EPOS-LHC'    : (806.1, 55.6,  0.15,  0.83, 3284, -260, 132, -0.462, -0.0008, 0.059)}

# ------------------  FUNCTIONS ----------------------
def gumbelParameters(lgE, A, model='EPOS-LHC'):
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
        'QGSJetII' : {
            'mu'     : ((758.444, -10.692, -1.253), (48.892, 0.02, 0.179), (-2.346, 0.348, -0.086)),
            'sigma'  : ((39.033, 7.452, -2.176), (4.390, -1.688, 0.170)),
            'lambda' : ((0.857, 0.686, -0.040), (0.179, 0.076, -0.0130))},
        'QGSJetII-04' : {
            'mu'     : ((761.383, -11.719, -1.372), (57.344, -1.731, 0.309), (-0.355, 0.273, -0.137)),
            'sigma'  : ((35.221, 12.335, -2.889), (0.307, -1.147, 0.271)),
            'lambda' : ((0.673, 0.694, -0.007), (0.060, -0.019, 0.017))},
        'Sibyll2.1' : {
            'mu'     : ((770.104, -15.873, -0.960), (58.668, -0.124, -0.023), (-1.423, 0.977, -0.191)),
            'sigma'  : ((31.717, 1.335, -0.601), (-1.912, 0.007, 0.086)),
            'lambda' : ((0.683, 0.278, 0.012), (0.008, 0.051, 0.003))},
        'EPOS1.99' : {
            'mu'     : ((780.013, -11.488, -1.906), (61.911, -0.098, 0.038), (-0.405, 0.163, -0.095)),
            'sigma'  : ((28.853, 8.104, -1.924), (-0.083, -0.961, 0.215)),
            'lambda' : ((0.538, 0.524, 0.047), (0.009, 0.023, 0.010))},
        'EPOS-LHC' : {
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

def gumbel(xmax, lgE, A, model='EPOS-LHC', scale=(1,1,1)):
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

def randGumbel(lgE, A, model='EPOS-LHC'):
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
        raise ValueError("Energy out of range log10(E/eV) = 17.8 - 20")
    return dXmax['energyBins'].searchsorted(lgE) - 1

def xmaxResolution(x, lgE):
    """
    Xmax resolution from [4]
    Returns: resolution pdf
    """
    i = getEnergyBin(lgE)
    s1, es1, s2, es2, k = dXmax['resolution'][i]

    g1 = normpdf(x, 0, s1)
    g2 = normpdf(x, 0, s2)
    return k * g1 + (1-k) * g2

def xmaxAcceptance(x, lgE):
    """
    Xmax acceptance from [4], equation (7)
    Returns: acceptance(x) between 0 - 1
    """
    i = getEnergyBin(lgE)
    x1, ex1, x2, ex2, l1, el1, l2, el2 = dXmax['acceptance'][i]

    x = np.array(x, dtype=float)
    lo = x < x1 # indices with Xmax < x1
    hi = x > x2 #              Xmax > x2
    acceptance = np.ones_like(x, )
    acceptance[lo] = np.exp( (x[lo] - x1) / l1)
    acceptance[hi] = np.exp(-(x[hi] - x2) / l2)
    return acceptance

def xmaxSystematics(lgE):
    """
    Systematic uncertainty on Xmax from [4]
    Returns Xhi, Xlo
    """
    i = getEnergyBin(lgE)
    return dXmax['systematics'][i]


def meanXmax(E, A, model='EPOS-LHC'):
    """
    <Xmax> values for given energies E [EeV], mass numbers A and a hadronic interaction model.
    See arXiv:1301.6637
    """
    X0, D, xi, delta = xmaxParams[model][:4]
    lE = np.log10(E) - 1
    return X0 + D*lE + (xi - D/np.log(10) + delta*lE)*np.log(A)

def varXmax(E, A, model='EPOS-LHC'):
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

def lnADistribution(E, A, weights=None, bins=dXmax['xmaxBins']):
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

def lnA2XmaxDistribution(lEc, mlnA, vlnA, model='EPOS-LHC'):
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

def xmaxDistribution(E, A, weights=None, model='EPOS-LHC', bins=dXmax['xmaxBins']):
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
def plotSpectrum(ax=None, scale=3, with_scale_uncertainty=False):
    """
    Plot the Auger spectrum.
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    logE = dSpectrum['logE']
    c = (10**logE)**scale
    J = c * dSpectrum['mean']
    Jhi = c * dSpectrum['stathi']
    Jlo = c * dSpectrum['statlo']

    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0,}
    ax.errorbar(logE, J, yerr=[Jlo, Jhi], fmt='ko', **kwargs)
    ax.errorbar(logE[:27], J[:27], yerr=[Jlo[:27], Jhi[:27]], fmt='ko', **kwargs)
    ax.plot(logE[27:], Jhi[27:], 'kv', **kwargs) # upper limits

    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel('E$^{%g}$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{%g}$]'%(scale, scale-1))
    ax.semilogy()

    # marker for the energy scale uncertainty
    if with_scale_uncertainty:
        uncertainty = np.array((0.86, 1.14))
        x = 20.25 + np.log10(uncertainty)
        y = uncertainty**scale * 1e38
        ax.plot(x, y, 'k', lw=0.8)
        ax.plot(20.25, 1e38, 'ko', ms=5)
        ax.text(20.25, 5e37, r'$\Delta E/E = 14\%$', ha='center', fontsize=12)

def plotMeanXmax(ax=None, models=['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04']):
    """
    Plot the Auger <Xmax> distribution.
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    d = dXmax['moments']
    lgE = d['meanLgEnergy']
    mX = d['meanXmax']
    stat = d['meanXmaxSigmaStat']
    syslo = d['meanXmaxSigmaSysLow']
    syshi = d['meanXmaxSigmaSysUp']

    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    l1 = ax.errorbar(lgE, mX, yerr=stat, fmt='ko', **kwargs)
    l2 = ax.errorbar(lgE, mX, yerr=[-syslo, syshi], fmt='', lw=0, mew=1.2, c='k', capsize=5)

    ax.set_xlim(17.5, 20)
    ax.set_ylim(640, 840)
    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g/cm$^2$]')

    previous_legend = ax.legend((l1, l2),
        ('data $\pm\sigma_\mathrm{stat}$', '$\pm\sigma_\mathrm{sys}$'),
        loc='upper left', fontsize=16, markerscale=0.8,
        handleheight=1.4, handlelength=0.8)

    if models:
        E = np.logspace(17.5, 20.5, 100) / 1e18
        A = np.ones(100)
        bins = np.linspace(17.5, 20.5, 101)
        ls = ('-', '--', ':')

        for i, m in enumerate(models):
            lE, mX1, vX1 = xmaxDistribution(E,    A, bins=bins, model=m)  # proton
            lE, mX2, vX2 = xmaxDistribution(E, 56*A, bins=bins, model=m)  # iron
            ax.plot(lE, mX1, 'k', lw=1, ls=ls[i], label=m)  # for legend
            ax.plot(lE, mX1, 'b', lw=1, ls=ls[i])
            ax.plot(lE, mX2, 'r', lw=1, ls=ls[i])

        ax.legend(loc='lower right', fontsize=14)
        ax.add_artist(previous_legend)

def plotStdXmax(ax=None, models=['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04']):
    """
    Plot the Auger sigma(Xmax) distribution.
    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    d = dXmax['moments']
    lgE = d['meanLgEnergy']
    sX = d['sigmaXmax']
    stat = d['sigmaXmaxSigmaStat']
    syslo = d['sigmaXmaxSigmaSysLow']
    syshi = d['sigmaXmaxSigmaSysUp']

    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    l1 = ax.errorbar(lgE, sX, yerr=stat, fmt='ko', **kwargs)
    l2 = ax.errorbar(lgE, sX, yerr=[-syslo, syshi], fmt='', lw=0, mew=1.2, c='k', capsize=5)

    ax.set_xlabel('$\log_{10}$($E$/eV)')
    ax.set_ylabel(r'$\sigma(\rm{X_{max}})$ [g/cm$^2$]')
    ax.set_xlim(17.5, 20)
    ax.set_ylim(1, 79)

    previous_legend = ax.legend((l1, l2),
        ('data $\pm\sigma_\mathrm{stat}$', '$\pm\sigma_\mathrm{sys}$'),
        loc='upper left', fontsize=16, markerscale=0.8,
        handleheight=1.4, handlelength=0.8)

    if models:
        E = np.logspace(17.5, 20.5, 100) / 1e18
        A = np.ones(100)
        bins = np.linspace(17.5, 20.5, 101)
        ls = ('-', '--', ':')

        for i, m in enumerate(models):
            lE, mX1, vX1 = xmaxDistribution(E,    A, bins=bins, model=m)  # proton
            lE, mX2, vX2 = xmaxDistribution(E, 56*A, bins=bins, model=m)  # iron
            ax.plot(lE, vX1**.5, 'k', lw=1, ls=ls[i], label=m)  # for legend
            ax.plot(lE, vX1**.5, 'b', lw=1, ls=ls[i])
            ax.plot(lE, vX2**.5, 'r', lw=1, ls=ls[i])

        ax.legend(loc='lower right', fontsize=14)
        ax.add_artist(previous_legend)

def plotSuper(scale=3, models=['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04']):
    """
    Plot spectrum and Xmax moments together
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,16))
    fig.subplots_adjust(hspace=0, wspace=0)
    ax1, ax2, ax3 = axes

    plotSpectrum(ax1, scale, True)
    plotMeanXmax(ax2, True, models)
    plotStdXmax(ax3, False, models)

    ax1.set_xlim(17.5, 20.5)
    ax1.set_ylim(1e36,2e38)

    # model description
    ax2.text(19, 825, 'proton', fontsize=16, rotation=22)
    ax2.text(20.2, 755, 'iron', fontsize=16, rotation=23)
    ax3.text(20.4, 59, 'proton', fontsize=16, ha='right')
    ax3.text(20.4, 12, 'iron', fontsize=16, ha='right')

    # ankle
    for ax in axes:
        ax.axvline(18.7, c='grey', lw=1)

    return fig, axes

def plotXmax(ax=None, i=0):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    h = dXmax['histograms'][i]
    bins = dXmax['xmaxBins']
    cens = dXmax['xmaxCens']
    ax.hist(bins, cens, weights=h, histtype='step', color='k', lw=1.5)

    ax.set_xlabel(r'$X_\mathrm{max}$ [g/cm$^2$]')
    ax.set_ylabel('N')

    Ebins = dXmax['energyBins']
    info  = '$\log_{10}(E) = %.1f - %.1f$' % (Ebins[i], Ebins[i+1])
    ax.text(0.98, 0.97, info, transform=ax.transAxes, ha='right', va='top', fontsize=14)

def plotXmaxAll():
    fig, axes = plt.subplots(6, 3, sharex=True, figsize=(12,20))
    axes = axes.flatten()
    for i in range(18):
        ax = axes[i]
        plotXmax(ax, i)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks((600, 800, 1000))
        # ax.locator_params(axis='y', nbins=5, min=0)

    axes[16].set_xlabel(r'$X_\mathrm{max}$ [g/cm$^2$]')
    axes[6].set_ylabel('events / (20 g/cm$^2$)')
