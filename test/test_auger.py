from astrotools import auger, stat
from pylab import *


nE = 20
lEbins = linspace(18, 20, nE+1)
lEcens = (lEbins[1:]+lEbins[:-1])/2
E = 10**(lEcens-18)
A = ones(nE)

def compare(model):
    lE, mX1, vX1 = auger.xmaxDistribution(E, A, bins=lEbins, model=model)

    mX2 = zeros(nE)
    sX2 = zeros(nE)
    for i in range(nE):
        X = auger.randXmax(ones(100000)*E[i], ones(100000)*A[i], model=model)
        mX2[i] = mean(X)
        sX2[i] = std(X)

    figure(figsize=(12, 6))
    subplot(121)
    plot(lEcens, mX1)
    plot(lEcens, mX2, label='gumbel')
    legend(loc='upper left', frameon=False)
    xlabel('$\log_{10}$(E/[eV])')
    ylabel(r'$\langle \rm{X_{max}} \rangle $ [g cm$^{-2}$]')

    subplot(122)
    plot(lEcens, vX1**.5)
    plot(lEcens, sX2)
    xlabel('$\log_{10}$(E/[eV])')
    ylabel(r'$\sigma(\rm{X_{max}})$ [g cm$^{-2}$]')

    suptitle(model)
    tight_layout()

compare('Epos 1.99')
compare('Epos-LHC')
compare('Sibyll 2.1')
compare('QGSJet II-04')
compare('QGSJet II')
show()
