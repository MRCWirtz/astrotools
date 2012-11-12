from matplotlib.pyplot import figure
from numpy import *

# ---------- Auger spectrum ----------
# Auger combined energy spectrum data from arXiv:1107.4809 (ICRC11)
lE_auger = array([18.05, 18.15, 18.25, 18.35, 18.45, 18.55, 18.65, 18.75, 18.85, 18.95, 19.05, 19.15, 19.25, 19.35, 19.45, 19.55, 19.65, 19.75, 19.85, 19.95, 20.05, 20.15, 20.25, 20.35, 20.45])
J_auger = array([3.03646e-17, 1.55187e-17, 6.9384e-18, 3.20104e-18, 1.52781e-18, 7.21632e-19, 3.62670e-19, 1.88793e-19, 1.01309e-19, 5.85637e-20, 3.27556e-20, 1.64764e-20, 8.27115e-21, 4.74636e-21, 2.13542e-21, 9.40782e-22, 3.18048e-22, 1.62447e-22, 6.92891e-23, 6.25398e-24, 3.20445e-24, 1.22847e-24, 0, 0, 0])
Jup_auger = array([1.16824e-18, 6.14107e-19, 3.08672e-19, 1.64533e-19, 9.72578e-21, 6.04038e-21, 3.91761e-21, 2.52816e-21, 1.66645e-21, 1.13438e-21, 7.56617e-22, 4.87776e-22, 3.01319e-22, 2.05174e-22, 1.24329e-22, 7.16586e-23, 3.77454e-23, 2.29493e-23, 1.30299e-23, 4.78027e-24, 3.59431e-24, 2.14480e-24, 1.19737e-24, 3.24772e-25, 8.79582e-26])
Jlo_auger = array([1.16824e-18, 6.14107e-19, 3.08672e-19, 1.64533e-19, 9.72578e-21, 6.04038e-21, 3.91761e-21, 2.52816e-21, 1.66645e-21, 1.13438e-21, 7.56617e-22, 4.87776e-22, 3.01319e-22, 2.05174e-22, 1.24329e-22, 7.16586e-23, 3.77454e-23, 2.29493e-23, 1.30299e-23, 3.92999e-24, 2.00836e-24, 7.69927e-25, 0, 0, 0])
# scale with E^3
c = (10**lE_auger)**3
J_auger *= c
Jup_auger *= c
Jlo_auger *= c


def plotSpectrum(E, weights=None, b=11):
  """
  Plots a given spectrum scaled to the Auger (ICRC 2011) spectrum
  
  Parameters
  ----------
  E : array
      energies [EeV]
  W : array, optional
      weights, do not need to be normalized
  b : int, optional
      bin number for the fit
  
  Returns
  -------
  figure : a matplotlib figure
  
  Examples
  --------
  from astrotools import auger
  from pylab import *
  E = logspace(0, 2, 1000)
  W = E**(-3)
  fig = auger.plotSpectrum(E, None, 7)
  show()
  """
  logEnergies = log10(E) + 18
  N, bins = histogram(logEnergies, weights=weights, bins=25, range=(18.0, 20.5))
  Nstd = N**.5

  # logarithmic bin centers
  lE = (bins[1:] + bins[:-1]) / 2

  # differential spectrum: divide by linear bin widths
  binWidths = 10**bins[1:] - 10**bins[:-1]
  J = N / binWidths
  Jup = J + Nstd / binWidths
  Jlo = J - Nstd / binWidths

  # scale with E^3
  c = (10**lE)**3
  J *= c
  Jup *= c
  Jlo *= c

  # scale to Auger spectrum in given bin
  c = J_auger[b] / J[b]
  J *= c
  Jup *= c
  Jlo *= c

  # ----------- Plotting ----------
  fig = figure()
  ax = fig.add_subplot(111)

  kwargs = {'linewidth':1, 'markersize':9, 'markeredgewidth':0}
  ax.plot(lE[b], J[b], 'go', ms=20, mfc=None, alpha=0.25, **kwargs)
  ax.errorbar(lE - 0.01, J, yerr=[J-Jlo, Jup-J], fmt='rs', **kwargs)
  ax.errorbar(lE_auger + 0.01, J_auger, yerr=[Jlo_auger, Jup_auger], uplims=J_auger==0, fmt='ko', **kwargs)

  ax.set_xlabel('$\log_{10}$(Energy/[eV])')
  ax.set_ylabel('E$^3$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^2$]')
  ax.set_ylim((1e36, 1e38))
  ax.semilogy()

  return fig


def plotSpectrumGroups(E, A, weights=None, b=11):
  """
  Plots a given spectrum and 4 elemental groups scaled to the Auger (ICRC 2011) spectrum
  
  Parameters
  ----------
  E : array
      energies [EeV]
  A : array
      mass numbers
  W : array, optional
      weights, do not need to be normalized
  b : int, optional
      bin number for the fit
  
  Returns
  -------
  figure : a matplotlib figure
  """
  idx1 = A == 1
  idx2 = (A >= 2) * (A <= 8)
  idx3 = (A >= 9) * (A <= 26)
  idx4 = (A >= 27)

  # spectrum (non-differential) with same bins as the Auger spectrum
  logEnergies = log10(E) + 18
  N, bins = histogram(logEnergies, weights=weights, bins=25, range=(18.0, 20.5))
  N1 = histogram(logEnergies[idx1], weights=weights[idx1], bins=25, range=(18.0, 20.5))[0]
  N2 = histogram(logEnergies[idx2], weights=weights[idx2], bins=25, range=(18.0, 20.5))[0]
  N3 = histogram(logEnergies[idx3], weights=weights[idx3], bins=25, range=(18.0, 20.5))[0]
  N4 = histogram(logEnergies[idx4], weights=weights[idx4], bins=25, range=(18.0, 20.5))[0]

  # logarithmic bin centers
  lE = (bins[1:] + bins[:-1]) / 2
  binWidths = 10**bins[1:] - 10**bins[:-1]

  # differential spectrum: divide by linear bin widths
  J = N / binWidths
  J1 = N1 / binWidths
  J2 = N2 / binWidths
  J3 = N3 / binWidths
  J4 = N4 / binWidths

  # scale with E^3
  c = (10**lE)**3
  J *= c
  J1 *= c
  J2 *= c
  J3 *= c
  J4 *= c

  # scale to Auger spectrum in given bin
  c = J_auger[b] / J[b]
  J *= c
  J1 *= c
  J2 *= c
  J3 *= c
  J4 *= c

  # ----- Plotting -----
  fig = figure()
  ax = fig.add_subplot(111)

  kwargs = {'linewidth':1, 'markersize':9, 'markeredgewidth':0}
  ax.errorbar(lE_auger + 0.01, J_auger, yerr=[Jlo_auger, Jup_auger], uplims=J_auger==0, fmt='ko', **kwargs)
  ax.plot(lE[b], J[b], 'go', ms=20, mfc=None, mew=0, alpha=0.25)
  ax.plot(lE, J, 'brown', label='Sum', **kwargs)
  ax.plot(lE, J1, 'b', label='p', **kwargs)
  ax.plot(lE, J2, 'gray', label='He', **kwargs)
  ax.plot(lE, J3, 'green', label='N', **kwargs)
  ax.plot(lE, J4, 'red', label='Fe', **kwargs)

  ax.set_xlabel('$\log_{10}$(Energy/[eV])')
  ax.set_ylabel('E$^3$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^2$]')
  ax.set_ylim((1e36, 1e38))
  ax.semilogy()

  return fig

