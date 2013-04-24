from astrotools import stat, coord
import numpy
from matplotlib.pyplot import figure
from StringIO import StringIO


# --------------------- DATA -------------------------
# Auger combined energy spectrum data from arXiv:1107.4809 (ICRC11)
# last 3 values are upper limits
dSpectrum = numpy.genfromtxt(
    StringIO("""#logE, mean, stathi, statlo
        18.05, 3.03646e-17, 1.16824e-18, 1.16824e-18
        18.15, 1.55187e-17, 6.14107e-19, 6.14107e-19
        18.25,  6.9384e-18, 3.08672e-19, 3.08672e-19
        18.35, 3.20104e-18, 1.64533e-19, 1.64533e-19
        18.45, 1.52781e-18, 9.72578e-21, 9.72578e-21
        18.55, 7.21632e-19, 6.04038e-21, 6.04038e-21
        18.65,  3.6267e-19, 3.91761e-21, 3.91761e-21
        18.75, 1.88793e-19, 2.52816e-21, 2.52816e-21
        18.85, 1.01309e-19, 1.66645e-21, 1.66645e-21
        18.95, 5.85637e-20, 1.13438e-21, 1.13438e-21
        19.05, 3.27556e-20, 7.56617e-22, 7.56617e-22
        19.15, 1.64764e-20, 4.87776e-22, 4.87776e-22
        19.25, 8.27115e-21, 3.01319e-22, 3.01319e-22
        19.35, 4.74636e-21, 2.05174e-22, 2.05174e-22
        19.45, 2.13542e-21, 1.24329e-22, 1.24329e-22
        19.55, 9.40782e-22, 7.16586e-23, 7.16586e-23
        19.65, 3.18048e-22, 3.77454e-23, 3.77454e-23
        19.75, 1.62447e-22, 2.29493e-23, 2.29493e-23
        19.85, 6.92891e-23, 1.30299e-23, 1.30299e-23
        19.95, 6.25398e-24, 4.78027e-24, 3.92999e-24
        20.05, 3.20445e-24, 3.59431e-24, 2.00836e-24
        20.15, 1.22847e-24,  2.1448e-24, 7.69927e-25
        20.25, 1.19737e-24,           0,           0
        20.35, 3.24772e-25,           0,           0
        20.45, 8.79582e-26,           0,           0"""),
    delimiter=',', names=True)

# Auger shower development data from arXiv:1107.4804 (ICRC11)
dXmax = numpy.genfromtxt(
    StringIO("""#logElo, logEhi, logE, meanlogE, N, mean, mstat, msyshi, msyslo, std, sstat, ssyshi, ssyslo
        18.0, 18.1, 18.05, 18.050, 1407, 713.8, 1.6, 10.0, 8.1, 55.4, 2.1, 5.3, 5.5
        18.1, 18.2, 18.15, 18.149, 1251, 722.0, 1.7, 10.1, 8.1, 56.1, 2.2, 5.2, 5.3
        18.2, 18.3, 18.25, 18.248,  998, 734.0, 1.9, 10.2, 8.1, 56.9, 2.3, 5.2, 5.2
        18.3, 18.4, 18.35, 18.349,  781, 736.7, 2.1, 10.3, 8.2, 54.1, 2.7, 5.1, 5.2
        18.4, 18.5, 18.45, 18.448,  619, 743.6, 2.4, 10.5, 8.3, 55.1, 3.3, 5.1, 5.1
        18.5, 18.6, 18.55, 18.547,  457, 746.9, 2.7, 10.6, 8.3, 53.5, 3.4, 5.1, 5.1
        18.6, 18.7, 18.65, 18.650,  331, 751.3, 3.0, 10.8, 8.5, 51.3, 4.2, 5.1, 5.1
        18.7, 18.8, 18.75, 18.750,  230, 749.8, 3.2, 11.0, 8.6, 44.6, 3.7, 5.1, 5.1
        18.8, 18.9, 18.85, 18.849,  188, 750.6, 3.6, 11.2, 8.7, 45.4, 3.6, 5.1, 5.1
        18.9, 19.0, 18.95, 18.946,  143, 756.5, 3.6, 11.3, 8.9, 38.7, 4.1, 5.1, 5.1
        19.0, 19.2, 19.10, 19.094,  186, 763.8, 3.3, 11.6, 9.1, 40.9, 3.9, 5.1, 5.1
        19.2, 19.4, 19.30, 19.289,  106, 766.2, 3.8, 12.0, 9.4, 34.9, 5.1, 5.1, 5.2
        19.4,    0, 19.55, 19.547,   47, 771.5, 4.7, 12.5, 9.7, 27.0, 6.0, 5.2, 5.4"""),
    delimiter=',', names=True)

# Auger mass composition data, parsed from arXiv:1301.6637
# Epos 1.99 <lnA>
dmA1 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 1.710508, 1.025054, 1.40379 , 1.472648, 1.334917
        18.15, 1.634344, 0.93953 , 1.321337, 1.393325, 1.255623
        18.25, 1.401698, 0.71    , 1.104385, 1.173228, 1.019877
        18.35, 1.541514, 0.843541, 1.244142, 1.328679, 1.156504
        18.45, 1.518514, 0.80808 , 1.211811, 1.299479, 1.108538
        18.55, 1.623945, 0.919624, 1.314053, 1.420485, 1.22017
        18.65, 1.707363, 0.978091, 1.391226, 1.507063, 1.278563
        18.75, 1.997445, 1.283733, 1.687508, 1.812735, 1.559196
        18.85, 2.218594, 1.454892, 1.886822, 2.027669, 1.742845
        18.95, 2.226953, 1.460106, 1.888936, 2.029797, 1.754334
        19.10, 2.303640, 1.518043, 1.965608, 2.087675, 1.831021
        19.30, 2.677257, 1.891469, 2.326646, 2.476912, 2.18267
        19.55, 3.907970, 2.258252, 2.737116, 2.909247, 2.552435"""),
    delimiter=',', names=True)

# Epos 1.99 sigma^2(lnA)
dvA1 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 2.548301, 0.42816, 1.544768, 1.876906, 1.212612
        18.15, 2.591069, 0.470953, 1.573426, 1.905598, 1.255405
        18.25, 2.47131, 0.280481, 1.439516, 1.778747, 1.100302
        18.35, 2.259703, 0.189048, 1.277371, 1.651937, 0.916955
        18.45, 2.394377, 0.267135, 1.390835, 1.821931, 0.952673
        18.55, 2.302903, 0.260492, 1.334697, 1.779926, 0.917736
        18.65, 2.119564, 0.147808, 1.186685, 1.695528, 0.670784
        18.75, 1.589919, -0.134462, 0.784257, 1.180009, 0.38143
        18.85, 1.851794, 0.162739, 1.067341, 1.463093, 0.657447
        18.95, 1.173739, -0.381057, 0.445815, 0.834499, 0.057114
        19.10, 1.44993, -0.133092, 0.714948, 1.10364, 0.312113
        19.30, 1.118547, -0.252494, 0.475429, 0.927718, 0.044334
        19.55, 0.688456, -0.414017, 0.172555, 0.596592, -0.230263"""),
    delimiter=',', names=True)

# Sibyll 2.1 <lnA>
dmA2 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 1.375736, 0.650177, 1.050648, 1.130742, 0.975265
        18.15, 1.281508, 0.56066, 0.961131, 1.036514, 0.881037
        18.25, 1.031802, 0.29682, 0.702002, 0.786808, 0.631331
        18.35, 1.154299, 0.424028, 0.824499, 0.928151, 0.744405
        18.45, 1.116608, 0.358068, 0.782097, 0.881037, 0.678445
        18.55, 1.215548, 0.471143, 0.881037, 0.998822, 0.767962
        18.65, 1.276796, 0.508834, 0.942285, 1.069494, 0.819788
        18.75, 1.573616, 0.829211, 1.234393, 1.361602, 1.116608
        18.85, 1.780919, 0.984688, 1.42285, 1.583039, 1.286219
        18.95, 1.780919, 0.975265, 1.42285, 1.573616, 1.276796
        19.10, 1.823322, 1.008245, 1.465253, 1.606596, 1.333333
        19.30, 2.195524, 1.366313, 1.823322, 1.98351, 1.672556
        19.55, 2.586572, 1.710247, 2.195524, 2.388693, 2.025913"""),
    delimiter=',', names=True)

# Sibyll 2.1 sigma^2(lnA)
dvA2 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 2.364857, -0.088724, 1.194487, 1.554055, 0.841945
        18.15, 2.392669, -0.004457, 1.236384, 1.617121, 0.876816
        18.25, 2.279497, -0.145841, 1.116136, 1.496864, 0.728366
        18.35, 2.046424, -0.322551, 0.91834, 1.341347, 0.516453
        18.45, 2.222289, -0.181938, 1.058945, 1.566599, 0.558366
        18.55, 2.137318, -0.217564, 0.995117, 1.516856, 0.48748
        18.65, 1.911279, -0.344854, 0.818457, 1.417743, 0.226213
        18.75, 1.276335, -0.725996, 0.310413, 0.789849, -0.16198
        18.85, 1.614378, -0.380927, 0.641407, 1.106749, 0.176056
        18.95, 0.782019, -1.044048, -0.092237, 0.358997, -0.536419
        19.10, 1.133982, -0.748498, 0.224465, 0.69685, -0.212675
        19.30, 0.752475, -0.883204, -0.037175, 0.484564, -0.530711
        19.55, 0.314332, -1.067553, -0.355467, 0.145128, -0.834886"""),
    delimiter=',', names=True)

# QGSJET 01 <lnA>
dmA3 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 0.890856, 0.137015, 0.556333, 0.631722, 0.476244
        18.15, 0.782732, 0, 0.415257, 0.50948, 0.339874
        18.25, 0.467332, 0, 0.128109, 0.222338, 0.029174
        18.35, 0.575962, 0, 0.222604, 0.316839, 0.123664
        18.45, 0.491412, 0, 0.128637, 0.246423, 0.029697
        18.55, 0.581201, 0, 0.204281, 0.326778, 0.0865
        18.65, 0.619165, 0, 0.237528, 0.383593, 0.100907
        18.75, 0.916247, 0.044636, 0.534615, 0.685386, 0.38856
        18.85, 1.104966, 0.223927, 0.723339, 0.883528, 0.553727
        18.95, 1.076941, 0.177062, 0.676475, 0.846087, 0.516269
        19.10, 1.110333, 0.153905, 0.691004, 0.851192, 0.540238
        19.30, 1.473608, 0.493651, 1.040161, 1.228619, 0.870549
        19.55, 1.884175, 0.819401, 1.422465, 1.658032, 1.196322"""),
    delimiter=',', names=True)

# QGSJET 01 sigma^2(lnA)
dvA3 = numpy.genfromtxt(StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 1.150588, -1.891765, -0.24, 0.12, -0.607059
        18.15, 1.072941, -2.047059, -0.388235, 0, -0.769412
        18.25, 0.649412, -1.898824, -0.889412, -0.465882, -1.305882
        18.35, 0.501176, -2.230588, -1.016471, -0.543529, -1.454118
        18.45, 0.543529, -2.054118, -1.023529, -0.458824, -1.567059
        18.55, 0.487059, -2.223529, -1.051765, -0.48, -1.616471
        18.65, 0.282353, -2.463529, -1.249412, -0.578824, -1.92
        18.75, -0.063529, -3, -1.461176, -0.903529, -2.025882
        18.85, 0.409412, -2.618824, -0.96, -0.388235, -1.524706
        18.95, -0.508235, -3, -1.828235, -1.249412, -2.385882
        19.10, -0.098824, -3, -1.552941, -0.974118, -2.103529
        19.30, -0.218824, -3, -1.524706, -0.896471, -2.167059
        19.55, -0.451765, -3, -1.602353, -0.967059, -2.251765"""),
    delimiter=',', names=True)

# QGSJET II <lnA>
dmA4 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 1.438679, 0.636792, 1.084906, 1.179245, 1.004717
        18.15, 1.29717, 0.495283, 0.933962, 1.023585, 0.84434
        18.25, 0.976415, 0.141509, 0.599057, 0.70283, 0.504717
        18.35, 1.066038, 0.231132, 0.693396, 0.792453, 0.589623
        18.45, 0.966981, 0.113208, 0.589623, 0.70283, 0.481132
        18.55, 1.028302, 0.160377, 0.650943, 0.783019, 0.523585
        18.65, 1.066038, 0.165094, 0.669811, 0.806604, 0.533019
        18.75, 1.330189, 0.457547, 0.95283, 1.113208, 0.806604
        18.85, 1.54717, 0.613208, 1.141509, 1.311321, 0.971698
        18.95, 1.504717, 0.542453, 1.080189, 1.25, 0.90566
        19.10, 1.5, 0.5, 1.066038, 1.226415, 0.910377
        19.30, 1.858491, 0.820755, 1.400943, 1.589623, 1.207547
        19.55, 2.226415, 1.136792, 1.75, 1.981132, 1.514151"""),
    delimiter=',', names=True)

# QGSJET II sigma^2(lnA)
dvA4 = numpy.genfromtxt(
    StringIO("""#logE, syshi, syslo, mean, stathi, statlo
        18.05, 2.795535, -0.243243, 1.350176, 1.780259, 0.927145
        18.15, 2.760282, -0.3349, 1.286722, 1.723854, 0.849589
        18.25, 2.485311, -0.73678, 0.920094, 1.371328, 0.454759
        18.35, 2.153937, -1.025852, 0.631022, 1.166863, 0.137485
        18.45, 2.231492, -1.068155, 0.673325, 1.307873, 0.052879
        18.55, 2.104583, -1.166863, 0.553467, 1.195065, -0.088132
        18.65, 1.773208, -1.484136, 0.250294, 1.011751, -0.511163
        18.75, 1.117509, -1.893067, -0.285546, 0.320799, -0.898942
        18.85, 1.53349, -1.455934, 0.144536, 0.764982, -0.46886
        18.95, 0.454759, -2.428907, -0.884841, -0.257344, -1.505288
        19.10, 0.849589, -2.231492, -0.59577, 0.038778, -1.216216
        19.30, 0.433608, -2.372503, -0.842538, -0.144536, -1.554642
        19.55, -0.102233, -2.647474, -1.230317, -0.532315, -1.942421"""),
    delimiter=',', names=True)

# Energy bin borders in log10(E/[eV]) used in Auger composition measurements
compositionBins = numpy.array([18,18.1,18.2,18.3,18.4,18.5,18.6,18.7,18.8,18.9,19,19.2,19.4,19.7])

# Values for <Xmax>, sigma(Xmax) parameterization, cf. arXiv:1301.6637 tables 1 and 2.
# xmaxParams[model] = [X0, D, xi, delta], [p0, p1, p2, a0, a1, b]
xmaxParams = {
    'Epos 1.99' : ([809.7, 62.2,  0.78,  0.08], [3279,  -47, 228, -0.461, -0.0041, 0.059]),
    'Sibyll 2.1': ([795.1, 57.7, -0.04, -0.04], [2785, -364, 152, -0.368, -0.0049, 0.039]),
    'QGSJet 01' : ([774.2, 49.7, -0.30,  1.92], [3852, -274, 169, -0.451, -0.0020, 0.057]),
    'QGSJet II' : ([781.8, 45.8, -1.13,  1.71], [3163, -237,  60, -0.386, -0.0006, 0.043])}

# Parameters for gumble - Xmax distribution, cf. GAP-2012-030
# Epos 1.99 values are from M. Domenico (private communication)
# gumbelParams[model]['mu'] = [a0, a1, a2], [b0, b1, b2]
gumbelParams = {
    'Epos 1.99' : {
        'mu'     : [[779.737, -11.4095, -1.96729], [62.3138, -0.29405, 0.139223]],
        'sigma'  : [[28.8526, 8.10448, -1.92435], [-0.0827813, -0.960622, 0.214701]],
        'lambda' : [[0.537829, 0.52411, 0.0468013], [0.00941438, 0.0233692, 0.00999622]]},
    'Sibyll 2.1': {
        'mu'     : [[768.880, -15.026, -1.125], [59.910, -0.980, 0.145]],
        'sigma'  : [[31.717, 1.335, -0.601], [-1.912, 0.007, 0.086]],
        'lambda' : [[0.683, 0.278, 0.012], [0.008, 0.051, 0.003]]},
    'QGSJet II' : {
        'mu'     : [[756.599, -10.322, -1.247], [50.998, -0.239, 0.246]],
        'sigma'  : [[39.033, 7.452, -2.176], [4.390, -1.688, 0.170]],
        'lambda' : [[0.857, 0.686, -0.040], [0.179, 0.076, -0.0130]]}
    }
# --------------------- DATA -------------------------


def geometricExposure(declination):
    """
    Auger geometric exposure for a given equatorial declination (see astro-ph/0004016).
    geometricExposure(declination (0,pi)) -> (0-1)
    """
     # convert from mathematical (0,pi) to astronomical (pi/2,-pi/2)
    declination = numpy.pi/2 - numpy.array(declination)
    if (abs(declination) > numpy.pi/2).any():
        raise Exception('geometricExposure: declination not in range (pi/2, -pi/2)')

    zmax = numpy.deg2rad(60.0)
    olat = numpy.deg2rad(-35.25)
    xi = (numpy.cos(zmax) - numpy.sin(olat) * numpy.sin(declination)) / (numpy.cos(olat) * numpy.cos(declination))
    xi = numpy.clip(xi, -1, 1)
    am = numpy.arccos(xi)
    exposure = numpy.cos(olat) * numpy.cos(declination) * numpy.sin(am) + am * numpy.sin(olat) * numpy.sin(declination)
    return exposure / 1.8131550872084088

def randDeclination(n=1):
    """
    Returns n random declinations (0,pi) drawn from the Auger exposure.
    """
    # estimate number of required trials, exposure is about 1/3 of the sky
    nTry = int(3.3 * n) + 50
    dec = numpy.arccos( numpy.random.rand(nTry) * 2 - 1 )
    accept = geometricExposure(dec) > numpy.random.rand(nTry)
    if sum(accept) < n:
        raise Exception("randDeclination: stochastical failure")
    return dec[accept][:n]

def randXmax(E, A, model='Epos 1.99'):
    """
    Random Xmax values for given energy E [EeV] and mass number A (cf. GAP-2012-030).
    """
    lE = numpy.log10(E/10.)
    lnA = numpy.log(A)
    D = numpy.array([numpy.ones(numpy.shape(E)), lnA, lnA**2])

    p = gumbelParams[model]
    a, b = p['mu']
    mu = numpy.dot(a, D) + numpy.dot(b, D) * lE
    a, b = p['sigma']
    sigma = numpy.dot(a, D) + numpy.dot(b, D) * lE
    a, b = p['lambda']
    lambd = numpy.dot(a, D) + numpy.dot(b, D) * lE

    # cf. Kragujevac J. Math. 25 (2003) 19-29, theorem 3.1:
    # Y = -ln X is generalized Gumbel distributed for Erlang distributed X
    # Erlang is a special case of the gamma distribution
    return mu - sigma * numpy.log( numpy.random.gamma(lambd, 1./lambd) )

def meanXmax(E, A, model='Epos 1.99'):
    """
    <Xmax> values for given energies E [EeV], mass numbers A and hadronic interaction model.
    See arXiv:1301.6637
    """
    X0, D, xi, delta = xmaxParams[model][0]
    lE = numpy.log10(E) - 1
    return X0 + D*lE + (xi - D/numpy.log(10) + delta*lE)*numpy.log(A)

def varXmax(E, A, model='Epos 1.99'):
    """
    sigma^2_sh(Xmax) values for given energies E [EeV], mass numbers A and hadronic interaction model.
    These are only the expected shower-to-shower fluctuations (eq. 2.8) for single energy and mass number.
    See arXiv:1301.6637
    """
    p0, p1, p2, a0, a1, b = xmaxParams[model][1]
    lE = numpy.log10(E) - 1
    lnA = numpy.log(A)
    s2p = p0 + p1*lE + p2*(lE**2)
    a = a0 + a1*lE
    return s2p*( 1 + a*lnA + b*(lnA**2) )

def xmaxDistribution(E, A, weights=None, model='Epos 1.99', bins=compositionBins):
    """
    Energy binned <Xmax> distribution for given energies E [EeV], mass numbers A, weights and hadronic interaction model.
    See arXiv:1301.6637
    """
    [X0, D, xi, delta], [p0, p1, p2, a0, a1, b] = xmaxParams[model]

    # all energies in log10(E / 10 EeV)
    lE = numpy.log10(E)-1
    lEbins = bins - 19
    lEcenter = (lEbins[1:] + lEbins[:-1])/2

    fE = (xi - D/numpy.log(10) + delta*lEcenter)
    s2p = p0 + p1*lEcenter + p2*(lEcenter**2)
    a = a0 + a1*lEcenter

    mlnA, vlnA = stat.binnedMeanAndVariance(lE, numpy.log(A), lEbins, weights=weights)
    mXmax = X0 + D*lEcenter + fE*mlnA # eq. 2.6
    vXmax = s2p*( 1 + a*mlnA + b*(vlnA + mlnA**2) ) + fE**2*vlnA # eq. 2.12
    return lEcenter+19, mXmax, vXmax

def lnADistribution(E, A, weights=None, bins=compositionBins):
    """
    Energy binned <lnA> and sigma^2(lnA) distribution for given energies E (EeV), mass numbers A and weights.
    """
    return stat.binnedMeanAndVariance(numpy.log10(E)+18, numpy.log(A), bins, weights)

def spectrum(E, weights=None, normalize2bin=None):
    """
    Differential spectrum for given energies [EeV] and optional weights.
    Optionally normalize to Auger spectrum in given bin.
    """
    N, bins = numpy.histogram(numpy.log10(E) + 18, weights=weights, bins=25, range=(18.0, 20.5))
    Nerr = N**.5 # poisson error

    binWidths = 10**bins[1:] - 10**bins[:-1] # linear bin widths
    J = N / binWidths # make differential
    Jerr = Nerr / binWidths

    if normalize2bin:
        c = dSpectrum['mean'][normalize2bin] / J[normalize2bin]
        J *= c
        Jerr *= c
    return lE, J, Jerr



# --------------------- PLOT -------------------------
def plotSpectrum(yList=None):
    """
    Plots a given spectrum scaled to the Auger (ICRC 2011) spectrum
    """
    logE = dSpectrum['logE']
    c = (10**logE)**3 # scale with E^3
    J = c * dSpectrum['mean']
    Jhi = c * dSpectrum['stathi']
    Jlo = c * dSpectrum['statlo']

    fig = figure()
    ax = fig.add_subplot(111)
    args = {'linewidth':1, 'markersize':8, 'markeredgewidth':0,}
    ax.errorbar(logE[:22], J[:22], yerr=[Jlo[:22], Jhi[:22]], fmt='ko', label='Auger (ICRC11)', **args)
    ax.plot(logE[22:], J[22:], 'kv', **args)

    if yList:
        for y in yList:
            ax.plot(logE, y * c)

    ax.set_xlabel('$\log_{10}$(E/[eV])')
    ax.set_ylabel('E$^3$ J(E) [km$^{-2}$ yr$^{-1}$ sr$^{-1}$ eV$^2$]')
    ax.set_ylim((1e36, 1e38))
    ax.semilogy()
    return fig

def plotMeanXmax(yList=None):
    """
    Plot the Auger <Xmax> distribution along with all distributions in xList.
    """
    fig = figure()
    ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(dXmax['logE'], dXmax['mean'], yerr=dXmax['mstat'], fmt='ko', **kwargs)
    ax.fill_between(dXmax['logE'], dXmax['mean']-dXmax['msyslo'], dXmax['mean']+dXmax['msyshi'], color='k', alpha=0.1)

    if yList != None:
        for y in yList:
            ax.plot(dXmax['logE'], y)

    ax.set_xlim(18, 20)
    ax.set_ylim(680, 830)
    ax.set_xlabel('$\log_{10}$(E/[eV])')
    ax.set_ylabel(r'$\langle \rm{X_{max}} \rangle $ [g cm$^{-2}$]')
    return fig

def plotStdXmax(yList=None):
    """
    Plot the Auger sigma(Xmax) distribution along with all distributions in xList.
    """
    fig = figure()
    ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(dXmax['logE'], dXmax['std'], yerr=dXmax['sstat'], fmt='ko', **kwargs)
    ax.fill_between(dXmax['logE'], dXmax['std']-dXmax['ssyslo'], dXmax['std']+dXmax['ssyshi'], color='k', alpha=0.1)

    if yList:
        for y in yList:
            ax.plot(dXmax['logE'], y)

    ax.set_xlim(18, 20)
    ax.set_ylim(0, 70)
    ax.set_xlabel('$\log_{10}$(E/[eV])')
    ax.set_ylabel(r'$\sigma(\rm{X_{max}})$ [g cm$^{-2}$]')
    return fig


def plotMeanLnA(yList=None, model='Epos 1.99'):
    """
    Plot Auger <lnA> distribution along with all distributions in xList.
    """
    mA = {'Epos 1.99' : dmA1,
          'Sibyll 2.1': dmA2,
          'QGSJet 01' : dmA3,
          'QGSJet II' : dmA4}[model]

    fig = figure()
    ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(mA['logE'], mA['mean'], yerr=[mA['mean']-mA['statlo'], mA['stathi']-mA['mean']], fmt='ko', **kwargs)
    ax.fill_between(mA['logE'], mA['syslo'], mA['syshi'], color='k', alpha=0.2)
    ax.text(18.2, 3.6, model)

    if yList:
        for y in yList:
            ax.plot(mA['logE'], y)

    ax.set_xlabel(r'$\log_{10}(E/\rm[eV])$')
    ax.set_ylabel(r'$\langle\ln A\rangle$')
    ax.set_xlim([18, 20])
    ax.set_ylim([0, 4])
    ax.set_yticks(numpy.arange(0, 4.5, 0.5))
    ax.set_yticklabels(['0', '', '1', '', '2', '', '3', '', '4'])
    return fig

def plotVarLnA(yList=None, model='Epos 1.99'):
    """
    Plot Auger sigma^2(lnA) distribution along with all distributions in xList.
    """
    vA = {'Epos 1.99' : dvA1,
          'Sibyll 2.1': dvA2,
          'QGSJet 01' : dvA3,
          'QGSJet II' : dvA4}[model]

    fig = figure()
    ax = fig.add_subplot(111)
    kwargs = {'linewidth':1, 'markersize':8, 'markeredgewidth':0}
    ax.errorbar(vA['logE'], vA['mean'], yerr=[vA['mean']-vA['statlo'], vA['stathi']-vA['mean']], fmt='ko', **kwargs)
    ax.fill_between(vA['logE'], vA['syslo'], vA['syshi'], color='k', alpha=0.2)
    ax.text(18.2, 3.6, model)

    if yList:
        for y in yList:
            ax.plot(vA['logE'], y)

    ax.set_xlabel(r'$\log_{10}(E/\rm[eV])$')
    ax.set_ylabel(r'$\sigma^2(\ln A)$')
    ax.set_xlim([18, 20])
    ax.set_ylim([0, 4])
    ax.set_yticks(numpy.arange(0, 4.5, 0.5))
    ax.set_yticklabels(['0', '', '1', '', '2', '', '3', '', '4'])
    return fig

