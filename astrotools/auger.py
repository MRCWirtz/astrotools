from matplotlib.pyplot import figure
from numpy import *
from StringIO import StringIO


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



# ---------- Auger composition ----------
# Auger composition from unpublished paper
# Epos 1.99 <lnA>
lnA1=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	1.710508	1.025054	1.40379	1.472648	1.334917
18.15	1.634344	0.93953	1.321337	1.393325	1.255623
18.25	1.401698	0.71	1.104385	1.173228	1.019877
18.35	1.541514	0.843541	1.244142	1.328679	1.156504
18.45	1.518514	0.80808	1.211811	1.299479	1.108538
18.55	1.623945	0.919624	1.314053	1.420485	1.22017
18.65	1.707363	0.978091	1.391226	1.507063	1.278563
18.75	1.997445	1.283733	1.687508	1.812735	1.559196
18.85	2.218594	1.454892	1.886822	2.027669	1.742845
18.95	2.226953	1.460106	1.888936	2.029797	1.754334
19.1	2.30364	1.518043	1.965608	2.087675	1.831021
19.3	2.677257	1.891469	2.326646	2.476912	2.18267
19.55	3.90797	2.258252	2.737116	2.909247	2.552435
""")
# Epos 1.99 sigma^2(lnA)
slnA1=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	2.548301	0.42816	1.544768	1.876906	1.212612
18.15	2.591069	0.470953	1.573426	1.905598	1.255405
18.25	2.47131	0.280481	1.439516	1.778747	1.100302
18.35	2.259703	0.189048	1.277371	1.651937	0.916955
18.45	2.394377	0.267135	1.390835	1.821931	0.952673
18.55	2.302903	0.260492	1.334697	1.779926	0.917736
18.65	2.119564	0.147808	1.186685	1.695528	0.670784
18.75	1.589919	-0.134462	0.784257	1.180009	0.38143
18.85	1.851794	0.162739	1.067341	1.463093	0.657447
18.95	1.173739	-0.381057	0.445815	0.834499	0.057114
19.1	1.44993	-0.133092	0.714948	1.10364	0.312113
19.3	1.118547	-0.252494	0.475429	0.927718	0.044334
19.55	0.688456	-0.414017	0.172555	0.596592	-0.230263""")
# Sibyll 2.1 <lnA>
lnA2=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	1.375736	0.650177	1.050648	1.130742	0.975265
18.15	1.281508	0.56066	0.961131	1.036514	0.881037
18.25	1.031802	0.29682	0.702002	0.786808	0.631331
18.35	1.154299	0.424028	0.824499	0.928151	0.744405
18.45	1.116608	0.358068	0.782097	0.881037	0.678445
18.55	1.215548	0.471143	0.881037	0.998822	0.767962
18.65	1.276796	0.508834	0.942285	1.069494	0.819788
18.75	1.573616	0.829211	1.234393	1.361602	1.116608
18.85	1.780919	0.984688	1.42285	1.583039	1.286219
18.95	1.780919	0.975265	1.42285	1.573616	1.276796
19.1	1.823322	1.008245	1.465253	1.606596	1.333333
19.3	2.195524	1.366313	1.823322	1.98351	1.672556
19.55	2.586572	1.710247	2.195524	2.388693	2.025913""")
# Sibyll 2.1 sigma^2(lnA)
slnA2=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	2.364857	-0.088724	1.194487	1.554055	0.841945
18.15	2.392669	-0.004457	1.236384	1.617121	0.876816
18.25	2.279497	-0.145841	1.116136	1.496864	0.728366
18.35	2.046424	-0.322551	0.91834	1.341347	0.516453
18.45	2.222289	-0.181938	1.058945	1.566599	0.558366
18.55	2.137318	-0.217564	0.995117	1.516856	0.48748
18.65	1.911279	-0.344854	0.818457	1.417743	0.226213
18.75	1.276335	-0.725996	0.310413	0.789849	-0.16198
18.85	1.614378	-0.380927	0.641407	1.106749	0.176056
18.95	0.782019	-1.044048	-0.092237	0.358997	-0.536419
19.1	1.133982	-0.748498	0.224465	0.69685	-0.212675
19.3	0.752475	-0.883204	-0.037175	0.484564	-0.530711
19.55	0.314332	-1.067553	-0.355467	0.145128	-0.834886""")
# QGSJET 01 <lnA>
lnA3=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	0.890856	0.137015	0.556333	0.631722	0.476244
18.15	0.782732	0	0.415257	0.50948	0.339874
18.25	0.467332	0	0.128109	0.222338	0.029174
18.35	0.575962	0	0.222604	0.316839	0.123664
18.45	0.491412	0	0.128637	0.246423	0.029697
18.55	0.581201	0	0.204281	0.326778	0.0865
18.65	0.619165	0	0.237528	0.383593	0.100907
18.75	0.916247	0.044636	0.534615	0.685386	0.38856
18.85	1.104966	0.223927	0.723339	0.883528	0.553727
18.95	1.076941	0.177062	0.676475	0.846087	0.516269
19.1	1.110333	0.153905	0.691004	0.851192	0.540238
19.3	1.473608	0.493651	1.040161	1.228619	0.870549
19.55	1.884175	0.819401	1.422465	1.658032	1.196322""")
# QGSJET 01 sigma^2(lnA)
slnA3=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	1.150588	-1.891765	-0.24	0.12	-0.607059
18.15	1.072941	-2.047059	-0.388235	0	-0.769412
18.25	0.649412	-1.898824	-0.889412	-0.465882	-1.305882
18.35	0.501176	-2.230588	-1.016471	-0.543529	-1.454118
18.45	0.543529	-2.054118	-1.023529	-0.458824	-1.567059
18.55	0.487059	-2.223529	-1.051765	-0.48	-1.616471
18.65	0.282353	-2.463529	-1.249412	-0.578824	-1.92
18.75	-0.063529	-3	-1.461176	-0.903529	-2.025882
18.85	0.409412	-2.618824	-0.96	-0.388235	-1.524706
18.95	-0.508235	-3	-1.828235	-1.249412	-2.385882
19.1	-0.098824	-3	-1.552941	-0.974118	-2.103529
19.3	-0.218824	-3	-1.524706	-0.896471	-2.167059
19.55	-0.451765	-3	-1.602353	-0.967059	-2.251765""")
# QGSJET II <lnA>
lnA4=StringIO("""18.05	1.438679	0.636792	1.084906	1.179245	1.004717
18.15	1.29717	0.495283	0.933962	1.023585	0.84434
18.25	0.976415	0.141509	0.599057	0.70283	0.504717
18.35	1.066038	0.231132	0.693396	0.792453	0.589623
18.45	0.966981	0.113208	0.589623	0.70283	0.481132
18.55	1.028302	0.160377	0.650943	0.783019	0.523585
18.65	1.066038	0.165094	0.669811	0.806604	0.533019
18.75	1.330189	0.457547	0.95283	1.113208	0.806604
18.85	1.54717	0.613208	1.141509	1.311321	0.971698
18.95	1.504717	0.542453	1.080189	1.25	0.90566
19.1	1.5	0.5	1.066038	1.226415	0.910377
19.3	1.858491	0.820755	1.400943	1.589623	1.207547
19.55	2.226415	1.136792	1.75	1.981132	1.514151""")
# QGSJET II sigma^2(lnA)
slnA4=StringIO("""#lg E	sys hi	sys lo	mean	stat hi	stat lo
18.05	2.795535	-0.243243	1.350176	1.780259	0.927145
18.15	2.760282	-0.3349	1.286722	1.723854	0.849589
18.25	2.485311	-0.73678	0.920094	1.371328	0.454759
18.35	2.153937	-1.025852	0.631022	1.166863	0.137485
18.45	2.231492	-1.068155	0.673325	1.307873	0.052879
18.55	2.104583	-1.166863	0.553467	1.195065	-0.088132
18.65	1.773208	-1.484136	0.250294	1.011751	-0.511163
18.75	1.117509	-1.893067	-0.285546	0.320799	-0.898942
18.85	1.53349	-1.455934	0.144536	0.764982	-0.46886
18.95	0.454759	-2.428907	-0.884841	-0.257344	-1.505288
19.1	0.849589	-2.231492	-0.59577	0.038778	-1.216216
19.3	0.433608	-2.372503	-0.842538	-0.144536	-1.554642
19.55	-0.102233	-2.647474	-1.230317	-0.532315	-1.942421""")


import ROOT
def plotComposition(E, A, weights=None):
  p = ROOT.TProfile('lnA-profile','',20,18.5,20.5)
  logEnergies = log10(E) + 18
  lnA = log(A)
  N = len(E)

  if weights == None:
    weights = zeros(N)

  for i in range(N):
    p.Fill(logEnergies[i], lnA[i], weights[i])

  x, y, yerr, n = zeros((4,20))
  for i in range(20):
    x[i] = p.GetBinCenter(i+1)
    y[i] = p.GetBinContent(i+1)
    yerr[i] = p.GetBinError(i+1)
    n[i] = p.GetBinEntries(i+1)

  x[y==0] = nan
  y[y==0] = nan
  yerr[y==0] = nan
  s2y = yerr**2 * n # get spread from mean error

  kwargs = {'linewidth':1, 'markersize':5, 'markeredgewidth':0}

  # ---------- Plot <lnA> ----------
  A1 = genfromtxt(lnA1, unpack=1)
  A2 = genfromtxt(lnA2, unpack=1)
  A3 = genfromtxt(lnA3, unpack=1)
  A4 = genfromtxt(lnA4, unpack=1)

  fig1 = figure()
  fig1.subplots_adjust(hspace=0)
  fig1.subplots_adjust(wspace=0)

  ax = fig1.add_subplot(111)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
  ax.set_xlabel(r'$\log_{10}(E/\rm[eV])$')
  ax.set_ylabel(r'$\langle\ln A\rangle$')

  ax1 = fig1.add_subplot(221)
  ax2 = fig1.add_subplot(222)
  ax3 = fig1.add_subplot(223)
  ax4 = fig1.add_subplot(224)

  ax1.set_xlim([18, 20.5])
  ax1.set_ylim([0, 4])
  ax2.set_xlim([18, 20.5])
  ax2.set_ylim([0, 4])
  ax3.set_xlim([18, 20.5])
  ax3.set_ylim([0, 4])
  ax4.set_xlim([18, 20.5])
  ax4.set_ylim([0, 4])

  ax1.set_yticks(arange(0, 4.5, 0.5))
  ax1.set_yticklabels(['0', '', '1', '', '2', '', '3', '', '4'])
  ax1.set_xticklabels([])

  ax2.set_xticklabels([])
  ax2.set_yticklabels([])

  ax3.set_xticks(arange(18, 21, 0.5))
  ax3.set_xticklabels(['18', '18.5', '19', '19.5', '20', ''])
  ax3.set_yticks(arange(0, 4.5, 0.5))
  ax3.set_yticklabels(['0', '', '1', '', '2', '', '3', '', ''])

  ax4.set_xticks(arange(18, 21, 0.5))
  ax4.set_xticklabels(['18', '18.5', '19', '19.5', '20', '20.5'])
  ax4.set_yticklabels([])

  ax1.fill_between(A1[0], A1[1], A1[2], color='k', alpha=0.2)
  ax1.errorbar(A1[0], A1[3], yerr=[A1[4]-A1[3], A1[3]-A1[5]], fmt='o', color='k', **kwargs)
  ax1.text(18.15, 3., 'Epos 1.99')
  ax1.errorbar(x, y, yerr=yerr, c='r', ecolor='r', fmt='s', **kwargs)

  ax2.fill_between(A2[0], A2[1], A2[2], color='k', alpha=0.2)
  ax2.errorbar(A2[0], A2[3], yerr=[A2[4]-A2[3], A2[3]-A2[5]], fmt='o', color='k', **kwargs)
  ax2.text(18.15, 3., 'Sibyll 2.1')
  ax2.errorbar(x, y, yerr=yerr, c='r', ecolor='r', fmt='s', **kwargs)

  ax3.fill_between(A3[0], A3[1], A3[2], color='k', alpha=0.2)
  ax3.errorbar(A3[0], A3[3], yerr=[A3[4]-A3[3], A3[3]-A3[5]], fmt='o', color='k', **kwargs)
  ax3.text(18.15, 3., 'QGSJET 01')
  ax3.errorbar(x, y, yerr=yerr, c='r', ecolor='r', fmt='s', **kwargs)

  ax4.fill_between(A4[0], A4[1], A4[2], color='k', alpha=0.2)
  ax4.errorbar(A4[0], A4[3], yerr=[A4[4]-A4[3], A4[3]-A4[5]], fmt='o', color='k', **kwargs)
  ax4.text(18.15, 3., 'QGSJET II')
  ax4.errorbar(x, y, yerr=yerr, c='r', ecolor='r', fmt='s', **kwargs)


  # ---------- Plot sigma(lnA) ----------
  B1 = genfromtxt(slnA1, unpack=1)
  B2 = genfromtxt(slnA2, unpack=1)
  B3 = genfromtxt(slnA3, unpack=1)
  B4 = genfromtxt(slnA4, unpack=1)

  fig2 = figure()
  fig2.subplots_adjust(hspace=0)
  fig2.subplots_adjust(wspace=0)

  ax = fig2.add_subplot(111)
  ax.spines['top'].set_color('none')
  ax.spines['bottom'].set_color('none')
  ax.spines['left'].set_color('none')
  ax.spines['right'].set_color('none')
  ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')
  ax.set_xlabel(r'$\log_{10}(E/\rm[eV])$')
  ax.set_ylabel(r'$\sigma^2(\ln A)$')

  ax1 = fig2.add_subplot(221)
  ax2 = fig2.add_subplot(222)
  ax3 = fig2.add_subplot(223)
  ax4 = fig2.add_subplot(224)

  ax1.set_xlim([18, 20.5])
  ax1.set_ylim([-2, 4])
  ax2.set_xlim([18, 20.5])
  ax2.set_ylim([-2, 4])
  ax3.set_xlim([18, 20.5])
  ax3.set_ylim([-2, 4])
  ax4.set_xlim([18, 20.5])
  ax4.set_ylim([-2, 4])

  ax1.set_xticklabels([])
  ax1.set_yticks(arange(-2, 5, 1))
  ax1.set_yticklabels(['', '-1', '0', '1', '2', '3'])

  ax2.set_xticklabels([])
  ax2.set_yticklabels([])

  ax3.set_xticks(arange(18, 21, 0.5))
  ax3.set_xticklabels(['18', '18.5', '19', '19.5', '20', ''])
  ax3.set_yticks(arange(-2, 5, 1))
  ax3.set_yticklabels(['', '-1', '0', '1', '2', '3'])
  #  ax3.set_yticks(arange(-3, 4, 1))
  #  ax3.set_yticklabels(['', '-2', '-1', '0', '1', '2', ''])

  ax4.set_xticks(arange(18, 21, 0.5))
  ax4.set_xticklabels(['18', '18.5', '19', '19.5', '20', '20.5'])
  ax4.set_yticklabels([])

  ax1.fill_between(B1[0], B1[1], B1[2], color='k', alpha=0.2)
  ax1.errorbar(B1[0], B1[3], yerr=[B1[4]-B1[3], B1[3]-B1[5]], fmt='o', color='k', **kwargs)
  ax1.text(18.15, 3, 'Epos 1.99')
  ax1.plot(x, s2y, 'rs', **kwargs)
  ax1.axhline(0, c='k', ls='--', lw=1)

  ax2.fill_between(B2[0], B2[1], B2[2], color='k', alpha=0.2)
  ax2.errorbar(B2[0], B2[3], yerr=[B2[4]-B2[3], B2[3]-B2[5]], fmt='o', color='k', **kwargs)
  ax2.text(18.15, 3, 'Sibyll 2.1')
  ax2.plot(x, s2y, 'rs', **kwargs)
  ax2.axhline(0, c='k', ls='--', lw=1)

  ax3.fill_between(B3[0], B3[1], B3[2], color='k', alpha=0.2)
  ax3.errorbar(B3[0], B3[3], yerr=[B3[4]-B3[3], B3[3]-B3[5]], fmt='o', color='k', **kwargs)
  ax3.text(18.15, 3, 'QGSJET 01')
  ax3.plot(x, s2y, 'rs', **kwargs)
  ax3.axhline(0, c='k', ls='--', lw=1)

  ax4.fill_between(B4[0], B4[1], B4[2], color='k', alpha=0.2)
  ax4.errorbar(B4[0], B4[3], yerr=[B4[4]-B4[3], B4[3]-B4[5]], fmt='o', color='k', **kwargs)
  ax4.text(18.15, 3, 'QGSJET II')
  ax4.plot(x, s2y, 'rs', **kwargs)
  ax4.axhline(0, c='k', ls='--', lw=1)

  return fig1, fig2



