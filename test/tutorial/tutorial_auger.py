import numpy as np
import matplotlib.pyplot as plt
from astrotools import auger

print("Test: module auger.py")

# Measured energy spectrum from ICRC 2017
dspectrum = auger.SPECTRA_DICT[17]
log10e = dspectrum['logE']
flux = (10 ** log10e) ** 3 * dspectrum['mean']
flux_high = (10 ** log10e) ** 3 * dspectrum['stathi']
flux_low = (10 ** log10e) ** 3 * dspectrum['statlo']
plt.errorbar(log10e[0:26], flux[:26], yerr=[flux_low[:26], flux_high[:26]],
             fmt='ko', linewidth=1, markersize=8, capsize=0, label='Data 2017')
yl = r'$E^{3} \, J(E)$ [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^{2}$]'
plt.ylabel(yl, fontsize=16)
plt.xlabel(r'$\log_{10}$($E$/eV)', fontsize=16)
# Analytic parametrization of AUGER energy spectrum
log10e = np.arange(17.5, 20.5, 0.02)
dN = auger.spectrum_analytic(log10e, year=19)
E3_dN = 10**(3 * log10e) * dN    # multiply with E^3 for better visability
plt.plot(log10e, E3_dN, color='red', label='Parametrization 2017')
plt.savefig('energy_spectrum.png')
plt.clf()

# We sample energies which follow the above parametrized energy spectrum
n, emin = 1e7, 18.5     # n: number of drawn samples; emin: 10 EeV; lower energy cut
log10e_sample = auger.rand_energy_from_auger(n=int(n), log10e_min=emin)
log10e_bins = np.arange(18.5, 20.55, 0.05)
n, bins = np.histogram(log10e_sample, bins=log10e_bins)
E3_dN_sampled = 10**((3-1)*(log10e_bins[:-1])) * n   # -1 for correcting logarithmic bin width

plt.plot(log10e[50:], E3_dN[50:], color='red')
plt.plot(log10e_bins[:-1], E3_dN_sampled * E3_dN[50] / E3_dN_sampled[0], marker='s', color='k', ls='None')
plt.yscale('log')
plt.xlabel('log10(E[eV])', fontsize=16)
plt.ylabel('E$^3$ dN', fontsize=16)
plt.savefig('energy_spectrum_sampled.png')
plt.clf()

##################################################
# Plot the Auger <Xmax> distribution
##################################################
fig = plt.figure()
ax = fig.add_subplot(111)

models = ['EPOS-LHC', 'Sibyll2.1', 'QGSJetII-04']
d = auger.DXMAX['moments17']

log10e = d['meanLgEnergy']
m_x = d['meanXmax']
e_stat = d['meanXmaxSigmaStat']
e_syslo = d['meanXmaxSigmaSysLow']
e_syshi = d['meanXmaxSigmaSysUp']

l1 = ax.errorbar(log10e, m_x, yerr=e_stat, fmt='ko', lw=1, ms=8, capsize=0)
l2 = ax.errorbar(log10e, m_x, yerr=[-e_syslo, e_syshi],
                 fmt='', lw=0, mew=1.2, c='k', capsize=5)

ax.set_xlim(17.5, 20)
ax.set_ylim(640, 840)
ax.set_xlabel(r'$\log_{10}$($E$/eV)')
ax.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g/cm$^2$]')

legend1 = ax.legend((l1, l2),
                    (r'data $\pm\sigma_\mathrm{stat}$',
                     r'$\pm\sigma_\mathrm{sys}$'),
                    loc='upper left', fontsize=16, markerscale=0.8,
                    handleheight=1.4, handlelength=0.8)

l_e = np.linspace(17.5, 20.5, 100)
ls = ('-', '--', ':')
for i, m in enumerate(models):
    m_x1 = auger.mean_xmax(l_e, 1, model=m)  # proton
    m_x2 = auger.mean_xmax(l_e, 56, model=m)  # iron
    ax.plot(l_e, m_x1, 'k', lw=1, ls=ls[i], label=m)  # for legend
    ax.plot(l_e, m_x1, 'r', lw=1, ls=ls[i])
    ax.plot(l_e, m_x2, 'b', lw=1, ls=ls[i])

ax.legend(loc='lower right', fontsize=14)
# noinspection PyUnboundLocalVariable
ax.add_artist(legend1)
plt.savefig('auger_xmax_mean.png', bbox_inches='tight')
plt.close()

##################################################
# Plot the Auger sigma(Xmax) distribution.
##################################################
fig = plt.figure()
ax = fig.add_subplot(111)

l_e = np.linspace(17.5, 20.5, 100)
ls = ('-', '--', ':')
for i, m in enumerate(models):
    v_x1 = auger.var_xmax(l_e, 1, model=m)  # proton
    v_x2 = auger.var_xmax(l_e, 56, model=m)  # iron
    ax.plot(l_e, v_x1 ** .5, 'k', lw=1, ls=ls[i], label=m)  # for legend
    ax.plot(l_e, v_x1 ** .5, 'r', lw=1, ls=ls[i])
    ax.plot(l_e, v_x2 ** .5, 'b', lw=1, ls=ls[i])

d = auger.DXMAX['moments17']
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
ax.set_xlim(17.0, 20)
ax.set_ylim(1, 79)

legend1 = ax.legend((l1, l2),
                    (r'data $\pm\sigma_\mathrm{stat}$',
                     r'$\pm\sigma_\mathrm{sys}$'),
                    loc='upper left', fontsize=16, markerscale=0.8,
                    handleheight=1.4, handlelength=0.8)
ax.legend(loc='lower right', fontsize=14)
ax.add_artist(legend1)
plt.savefig('auger_xmax_std.png', bbox_inches='tight')
plt.close()
