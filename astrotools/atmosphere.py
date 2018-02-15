"""
Atmospheric density models as used in CORSIKA.
The parameters are documented in the CORSIKA manual
The parameters for the Auger atmospheres are documented in detail in GAP2011-133
The May and October atmospheres describe the annual average best.
Parameters

- a in g/cm^2 --> g/m^2
- b in g/cm^2 --> g/m^2
- c in cm --> m
- h in km --> m

"""
import os.path

import numpy as np
from scipy import integrate, interpolate, optimize

R_E = 6.371 * 1e6  # radius of Earth
H_MAX = 112829.2  # height above sea level where the mass overburden vanishes

DEFAULT_MODEL = 17
ATM_MODELS = {
    1: {  # US standard after Linsley
        'a': 1e4 * np.array([-186.555305, -94.919, 0.61289, 0., 0.01128292]),
        'b': 1e4 * np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.]),
        'c': 1e-2 * np.array([994186.38, 878153.55, 636143.04, 772170.16, 1.e9]),
        'h': 1e3 * np.array([0., 4., 10., 40., 100.])},
    17: {  # US standard after Keilhauer
        'a': 1e4 * np.array([-149.801663, -57.932486, 0.63631894, 4.35453690e-4, 0.01128292]),
        'b': 1e4 * np.array([1183.6071, 1143.0425, 1322.9748, 655.67307, 1.]),
        'c': 1e-2 * np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.e9]),
        'h': 1e3 * np.array([0., 7., 11.4, 37., 100.])},
    18: {  # Malargue January
        'a': 1e4 * np.array([-136.72575606, -31.636643044, 1.8890234035, 3.9201867984e-4, 0.01128292]),
        'b': 1e4 * np.array([1174.8298334, 1204.8233453, 1637.7703583, 735.96095023, 1.]),
        'c': 1e-2 * np.array([982815.95248, 754029.87759, 594416.83822, 733974.36972, 1e9]),
        'h': 1e3 * np.array([0., 9.4, 15.3, 31.6, 100.])},
    19: {  # Malargue February
        'a': 1e4 * np.array([-137.25655862, -31.793978896, 2.0616227547, 4.1243062289e-4, 0.01128292]),
        'b': 1e4 * np.array([1176.0907565, 1197.8951104, 1646.4616955, 755.18728657, 1.]),
        'c': 1e-2 * np.array([981369.6125, 756657.65383, 592969.89671, 731345.88332, 1.e9]),
        'h': 1e3 * np.array([0., 9.2, 15.4, 31., 100.])},
    20: {  # Malargue March
        'a': 1e4 * np.array([-132.36885162, -29.077046629, 2.090501509, 4.3534337925e-4, 0.01128292]),
        'b': 1e4 * np.array([1172.6227784, 1215.3964677, 1617.0099282, 769.51991638, 1.]),
        'c': 1e-2 * np.array([972654.0563, 742769.2171, 595342.19851, 728921.61954, 1.e9]),
        'h': 1e3 * np.array([0., 9.6, 15.2, 30.7, 100.])},
    21: {  # Malargue April
        'a': 1e4 * np.array([-129.9930412, -21.847248438, 1.5211136484, 3.9559055121e-4, 0.01128292]),
        'b': 1e4 * np.array([1172.3291878, 1250.2922774, 1542.6248413, 713.1008285, 1.]),
        'c': 1e-2 * np.array([962396.5521, 711452.06673, 603480.61835, 735460.83741, 1.e9]),
        'h': 1e3 * np.array([0., 10., 14.9, 32.6, 100.])},
    22: {  # Malargue May
        'a': 1e4 * np.array([-125.11468467, -14.591235621, 0.93641128677, 3.2475590985e-4, 0.01128292]),
        'b': 1e4 * np.array([1169.9511302, 1277.6768488, 1493.5303781, 617.9660747, 1.]),
        'c': 1e-2 * np.array([947742.88769, 685089.57509, 609640.01932, 747555.95526, 1.e9]),
        'h': 1e3 * np.array([0., 10.2, 15.1, 35.9, 100.])},
    23: {  # Malargue June
        'a': 1e4 * np.array([-126.17178851, -7.7289852811, 0.81676828638, 3.1947676891e-4, 0.01128292]),
        'b': 1e4 * np.array([1171.0916276, 1295.3516434, 1455.3009344, 595.11713507, 1.]),
        'c': 1e-2 * np.array([940102.98842, 661697.57543, 612702.0632, 749976.26832, 1.e9]),
        'h': 1e3 * np.array([0., 10.1, 16., 36.7, 100.])},
    24: {  # Malargue July
        'a': 1e4 * np.array([-126.17216789, -8.6182537514, 0.74177836911, 2.9350702097e-4, 0.01128292]),
        'b': 1e4 * np.array([1172.7340688, 1258.9180079, 1450.0537141, 583.07727715, 1.]),
        'c': 1e-2 * np.array([934649.58886, 672975.82513, 614888.52458, 752631.28536, 1.e9]),
        'h': 1e3 * np.array([0., 9.6, 16.5, 37.4, 100.])},
    25: {  # Malargue August
        'a': 1e4 * np.array([-123.27936204, -10.051493041, 0.84187346153, 3.2422546759e-4, 0.01128292]),
        'b': 1e4 * np.array([1169.763036, 1251.0219808, 1436.6499372, 627.42169844, 1.]),
        'c': 1e-2 * np.array([931569.97625, 678861.75136, 617363.34491, 746739.16141, 1.e9]),
        'h': 1e3 * np.array([0., 9.6, 15.9, 36.3, 100.])},
    26: {  # Malargue September
        'a': 1e4 * np.array([-126.94494665, -9.5556536981, 0.74939405052, 2.9823116961e-4, 0.01128292]),
        'b': 1e4 * np.array([1174.8676453, 1251.5588529, 1440.8257549, 606.31473165, 1.]),
        'c': 1e-2 * np.array([936953.91919, 678906.60516, 618132.60561, 750154.67709, 1.e9]),
        'h': 1e3 * np.array([0., 9.5, 15.9, 36.3, 100.])},
    27: {  # Malargue October
        'a': 1e4 * np.array([-133.13151125, -13.973209265, 0.8378263431, 3.111742176e-4, 0.01128292]),
        'b': 1e4 * np.array([1176.9833473, 1244.234531, 1464.0120855, 622.11207419, 1.]),
        'c': 1e-2 * np.array([954151.404, 692708.89816, 615439.43936, 747969.08133, 1.e9]),
        'h': 1e3 * np.array([0., 9.5, 15.5, 36.5, 100.])},
    28: {  # Malargue November
        'a': 1e4 * np.array([-134.72208165, -18.172382908, 1.1159806845, 3.5217025515e-4, 0.01128292]),
        'b': 1e4 * np.array([1175.7737972, 1238.9538504, 1505.1614366, 670.64752105, 1.]),
        'c': 1e-2 * np.array([964877.07766, 706199.57502, 610242.24564, 741412.74548, 1.e9]),
        'h': 1e3 * np.array([0., 9.6, 15.3, 34.6, 100.])},
    29: {  # Malargue December
        'a': 1e4 * np.array([-135.40825209, -22.830409026, 1.4223453493, 3.7512921774e-4, 0.01128292]),
        'b': 1e4 * np.array([1174.644971, 1227.2753683, 1585.7130562, 691.23389637, 1.]),
        'c': 1e-2 * np.array([973884.44361, 723759.74682, 600308.13983, 738390.20525, 1.e9]),
        'h': 1e3 * np.array([0., 9.6, 15.6, 33.3, 100.])}}


def distance2height(d, zenith, observation_level=0):
    """Height above ground for given distance and zenith angle"""
    r = R_E + observation_level
    x = d * np.sin(zenith)
    y = d * np.cos(zenith) + r
    h = (x ** 2 + y ** 2) ** 0.5 - r
    return h


def height2distance(h, zenith, observation_level=0):
    """Distance for given height above ground and zenith angle"""
    r = R_E + observation_level
    return (h ** 2 + 2 * r * h + r ** 2 * np.cos(zenith) ** 2) ** 0.5 - r * np.cos(zenith)


def height2overburden(h, model=DEFAULT_MODEL):
    """
    Amount of atmosphere above given height.

    :param h: height above sea level in meter
    :param model: atmospheric model, default is 17 (US standard after Keilhauer)
    :return: atmospheric overburden in g/cm^2
    """
    a = ATM_MODELS[model]['a']
    b = ATM_MODELS[model]['b']
    c = ATM_MODELS[model]['c']
    layers = ATM_MODELS[model]['h']
    h = np.array(h)
    i = layers.searchsorted(h) - 1
    i = np.clip(i, 0, None)  # use layer 0 for negative heights
    x = np.where(i < 4,
                 a[i] + b[i] * np.exp(-h / c[i]),  # pylint: disable=E1130
                 a[4] - b[4] * h / c[4])
    x = np.where(h > H_MAX, 0, x)
    return x * 1E-4


def overburden2height(x, model=DEFAULT_MODEL):
    """
    Height for given overburden

    :param x: atmospheric overburden in g/cm^2
    :param model: atmospheric model, default is 17 (US standard after Keilhauer)
    :return: height above sea level in meter
    """
    a = ATM_MODELS[model]['a']
    b = ATM_MODELS[model]['b']
    c = ATM_MODELS[model]['c']
    layers = ATM_MODELS[model]['h']
    xlayers = height2overburden(layers, model=model)
    x = np.array(x)
    i = xlayers.size - np.searchsorted(xlayers[::-1], x) - 1
    i = np.clip(i, 0, None)
    h = np.where(i < 4,
                 -c[i] * np.log((x * 1E4 - a[i]) / b[i]),
                 -c[4] * (x * 1E4 - a[4]) / b[4])
    h = np.where(x <= 0, H_MAX, h)
    return h


def density(h, model=DEFAULT_MODEL):
    """
    Atmospheric density at given height

    :param h: height above sea level in m
    :param model: atmospheric model, default is 17 (US standard after Keilhauer)
    :return: atmospheric overburden in g/m^3
    """
    h = np.array(h)

    if model == 'barometric':  # barometric formula
        r = 8.31432  # universal gas constant for air: 8.31432 N m/(mol K)
        g0 = 9.80665  # gravitational acceleration (9.80665 m/s2)
        m = 0.0289644  # molar mass of Earth's air (0.0289644 kg/mol)
        rb = [1.2250, 0.36391, 0.08803, 0.01322, 0.00143, 0.00086, 0.000064]
        tb = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
        lb = [-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002]
        hb = [0, 11000, 20000, 32000, 47000, 51000, 71000]

        def rho1(_h, _i):
            """for lb == 0"""
            return rb[_i] * np.exp(-g0 * m * (_h - hb[_i]) / (r * tb[_i]))

        def rho2(_h, _i):
            """for lb != 0"""
            return rb[_i] * (tb[_i] / (tb[_i] + lb[_i] * (_h - hb[_i]))) ** (1 + (g0 * m) / (r * lb[_i]))

        i = np.searchsorted(hb, h) - 1
        rho = np.where(lb[i] == 0, rho1(h, i), rho2(h, i))
        rho = np.where(h > 86000, 0, rho)
        return rho * 1e3

    b = ATM_MODELS[model]['b']
    c = ATM_MODELS[model]['c']
    layers = ATM_MODELS[model]['h']
    i = np.searchsorted(layers, h) - 1
    rho = np.where(i < 4, np.exp(-h / c[i]), 1) * b[i] / c[i]  # pylint: disable=E1130
    return rho


def refractive_index(h, n0=1.000292, model=DEFAULT_MODEL):
    """Refractive index at given height.

    Args:
        h (array): height above sea level in [m]
        n0 (float, optional): refractive index at sea level
        model (int, optional): atmospheric model

    Returns:
        array: refractive index at given height
    """
    return 1 + (n0 - 1) * density(h, model) / density(0, model)


class Atmosphere:
    """Atmosphere class from radiotools.
    Could use some refactoring.
    For reference see PhD C. Glaser, appendix
    """

    def __init__(self, model=17, n_taylor=5, curved=True, zenith_numeric=np.deg2rad(83), filename=None):
        print('Using model %i' % model)
        self.model = model
        self.curved = curved
        self.n_taylor = n_taylor
        self.zenith_numeric = 0
        self.b = ATM_MODELS[model]['b']
        self.c = ATM_MODELS[model]['c']
        self.h = ATM_MODELS[model]['h']
        self.zenith = np.arccos(np.linspace(0, 1, 101))

        if not curved:
            return

        if filename is None:
            filename = 'atmosphere_model%i.npz' % model

        if os.path.exists(filename):
            print('Reading constants from %s' % filename)
            data = np.load(filename)
            assert self.model == data['model'], 'File contains parameters for different model %i' % model
            self.a = data['a']
        else:
            print('Calculating constants for curved atmosphere')
            self.a = self.__calculate_a()
            np.savez_compressed(filename, a=self.a, model=model)

        mask = self.zenith < zenith_numeric
        self.a_funcs = []
        for i in range(5):
            func = interpolate.interp1d(self.zenith[mask], self.a[:, i][mask], kind='cubic')
            self.a_funcs.append(func)

    def __calculate_a(self, ):
        b = self.b
        c = self.c
        h = self.h
        a = np.zeros((len(self.zenith), 5))
        for i, z in enumerate(self.zenith):
            print("zenith %.02f" % np.rad2deg(z))
            x_layers = [self._get_atmosphere_numeric([z], h_low=hh) for hh in h]
            dldh = [self._get_dldh(h[i], z, i) for i in range(5)]
            a[i, 0] = x_layers[0] - b[0] * dldh[0]
            a[i, 1] = x_layers[1] - b[1] * np.exp(-h[1] / c[1]) * dldh[1]
            a[i, 2] = x_layers[2] - b[2] * np.exp(-h[2] / c[2]) * dldh[2]
            a[i, 3] = x_layers[3] - b[3] * np.exp(-h[3] / c[3]) * dldh[3]
            a[i, 4] = x_layers[4] + b[4] * h[4] / c[4] * dldh[4]
        return a

    def _get_dldh(self, h, zenith, i):
        if i < 4:
            c = self.c[i]
            st = np.sin(zenith)
            ct = np.cos(zenith)
            dldh = np.ones_like(zenith) / ct
            if self.n_taylor >= 1:
                dldh += -(st ** 2 / ct ** 3 * (c + h) / R_E)
            if self.n_taylor >= 2:
                dldh += 1.5 * st ** 2 * (2 * c ** 2 + 2 * c * h + h ** 2) / (R_E ** 2 * ct ** 5)
            if self.n_taylor >= 3:
                t1 = 6 * c ** 3 + 6 * c ** 2 * h + 3 * c * h ** 2 + h ** 3
                dldh += st ** 2 / (2 * R_E ** 3 * ct ** 7) * (ct ** 2 - 5) * t1
            if self.n_taylor >= 4:
                t1 = 24 * c ** 4 + 24 * c ** 3 * h + 12 * c ** 2 * h ** 2 + 4 * c * h ** 3 + h ** 4
                dldh += -1. * st ** 2 * 5. / (8. * R_E ** 4 * ct ** 9) * (3 * ct ** 2 - 7) * t1
            if self.n_taylor >= 5:
                t1 = 120 * c ** 5 + 120 * c ** 4 * h + 60 * c ** 3 * h ** 2
                t1 += 20 * c ** 2 * h ** 3 + 5 * c * h ** 4 + h ** 5
                dldh += st ** 2 * (ct ** 4 - 14. * ct ** 2 + 21.) * (-3. / 8.) / (R_E ** 5 * ct ** 11) * t1
        elif i == 4:
            st = np.sin(zenith)
            ct = np.cos(zenith)
            dldh = np.ones_like(zenith) / ct
            if self.n_taylor >= 1:
                dldh += (-0.5 * st ** 2 / ct ** 3 * h / R_E)
            if self.n_taylor >= 2:
                dldh += 0.5 * st ** 2 / ct ** 5 * (h / R_E) ** 2
            if self.n_taylor >= 3:
                dldh += 1. / 8. * (st ** 2 * (ct ** 2 - 5) * h ** 3) / (R_E ** 3 * ct ** 7)
            if self.n_taylor >= 4:
                dldh += -1. / 8. * st ** 2 * (3 * ct ** 2 - 7) * (h / R_E) ** 4 / ct ** 9
            if self.n_taylor >= 5:
                dldh += -1. / 16. * st ** 2 * (ct ** 4 - 14 * ct ** 2 + 21) * (h / R_E) ** 5 / ct ** 11
        else:
            raise ValueError("ERROR, height index our of bounds")

        return dldh

    @staticmethod
    def __get_arguments(mask, *args):
        """Helper function"""
        tmp = []
        for arg in args:
            if np.shape(arg) == ():
                tmp.append(arg * np.ones(np.array(mask).size))
            else:
                tmp.append(arg[mask])
        return tmp

    def __get_method_mask(self, zenith):
        if not self.curved:
            return np.ones_like((3, zenith), dtype=np.bool)
        mask_flat = np.zeros_like(zenith, dtype=np.bool)
        mask_taylor = zenith < self.zenith_numeric
        mask_numeric = zenith >= self.zenith_numeric
        return mask_flat, mask_taylor, mask_numeric

    def __get_height_masks(self, h):
        layers = self.h
        mask0 = (h < layers[0])
        mask1 = (h >= layers[0]) & (h < layers[1])
        mask2 = (h >= layers[1]) & (h < layers[2])
        mask3 = (h >= layers[2]) & (h < layers[3])
        mask4 = (h >= layers[3]) & (h < H_MAX)
        mask5 = h >= H_MAX
        return np.array([mask0, mask1, mask2, mask3, mask4, mask5])

    def __get_x_masks(self, x, zenith):
        layers = [self._get_atmosphere(zenith, h) for h in self.h]
        mask0 = x > layers[0]
        mask1 = (x <= layers[0]) & (x > layers[1])
        mask2 = (x <= layers[1]) & (x > layers[2])
        mask3 = (x <= layers[2]) & (x > layers[3])
        mask4 = (x <= layers[3]) & (x > self._get_atmosphere(zenith, H_MAX))
        mask5 = x <= 0
        return np.array([mask0, mask1, mask2, mask3, mask4, mask5])

    def get_atmosphere(self, zenith, h_low=0., h_up=np.infty):
        """ returns the atmosphere for an air shower with given zenith angle (in g/cm^2) """
        return self._get_atmosphere(zenith, h_low=h_low, h_up=h_up) * 1e-4

    def _get_atmosphere(self, zenith, h_low=0., h_up=np.infty):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        mask_finite = np.array((h_up * np.ones_like(zenith)) < H_MAX)
        is_mask_finite = np.sum(mask_finite)
        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            args = self.__get_arguments(mask_numeric, zenith, h_low, h_up)
            tmp[mask_numeric] = self._get_atmosphere_numeric(*args)  # pylint: disable=E1120
        if np.sum(mask_taylor):
            args = self.__get_arguments(mask_taylor, zenith, h_low)
            tmp[mask_taylor] = self._get_atmosphere_taylor(*args)  # pylint: disable=E1120
            if is_mask_finite:
                mask_tmp = np.squeeze(mask_finite[mask_taylor])
                args = self.__get_arguments(mask_taylor, zenith, h_up)
                tmp2 = self._get_atmosphere_taylor(*args)  # pylint: disable=E1120
                tmp[mask_tmp] = tmp[mask_tmp] - np.array(tmp2)
        if np.sum(mask_flat):
            args = self.__get_arguments(mask_flat, zenith, h_low)
            tmp[mask_flat] = self._get_atmosphere_flat(*args)  # pylint: disable=E1120
            if is_mask_finite:
                mask_tmp = np.squeeze(mask_finite[mask_flat])
                args = self.__get_arguments(mask_flat, zenith, h_up)
                tmp2 = self._get_atmosphere_flat(*args)  # pylint: disable=E1120
                tmp[mask_tmp] = tmp[mask_tmp] - np.array(tmp2)
        return tmp

    def _get_atmosphere_taylor(self, zenith, h_low=0.):
        b = self.b
        c = self.c
        a = np.c_[[self.a_funcs[i](zenith) for i in range(5)]]

        masks = self.__get_height_masks(h_low)
        tmp = np.zeros_like(zenith)
        for i, mask in enumerate(masks):
            if np.sum(mask):
                if np.array(h_low).size == 1:
                    h = h_low
                else:
                    h = h_low[mask]
                if i < 4:
                    dldh = self._get_dldh(h, zenith[mask], i)
                    tmp[mask] = np.array([a[..., i][mask] + b[i] * np.exp(-1 * h / c[i]) * dldh])
                elif i == 4:
                    dldh = self._get_dldh(h, zenith[mask], i)
                    tmp[mask] = np.array([a[..., i][mask] - b[i] * h / c[i] * dldh])
                else:
                    tmp[mask] = np.zeros(np.sum(mask))
        return tmp

    def _get_vertical_height_numeric(self, zenith, x):
        tmp = np.zeros_like(zenith)
        zenith = np.array(zenith)
        for i, _ in enumerate(tmp):
            x0 = height2distance(self._get_vertical_height_flat(zenith[i], x[i]), zenith[i])

            def ftmp(d, zen, xmax, observation_level=0):
                """Internal helper function"""
                h = distance2height(d, zen, observation_level=observation_level)
                h += observation_level
                tmp = self._get_atmosphere_numeric([zen], h_low=h)
                dtmp = tmp - xmax
                return dtmp

            dxmax_geo = optimize.brentq(ftmp, -1e3, x0 + 1e4, xtol=1e-6, args=(zenith[i], x[i]))
            tmp[i] = distance2height(dxmax_geo, zenith[i])
        return tmp

    def _get_atmosphere_numeric(self, zenith, h_low=0, h_up=np.infty):
        zenith = np.array(zenith)
        tmp = np.zeros_like(zenith)
        for i, _ in enumerate(tmp):
            if np.array(h_up).size == 1:
                t_h_up = h_up
            else:
                t_h_up = h_up[i]
            if np.array(h_low).size == 1:
                t_h_low = h_low
            else:
                t_h_low = h_low[i]
            z = zenith[i]
            if t_h_up <= t_h_low:
                return np.nan
            if t_h_up == np.infty:
                t_h_up = H_MAX
            b = t_h_up
            d_low = height2distance(t_h_low, z)
            d_up = height2distance(b, z)
            full_atm = integrate.quad(self._get_density4, d_low, d_up, args=(z,), limit=500)[0]
            tmp[i] = full_atm
        return tmp

    def _get_atmosphere_flat(self, zenith, h=0):
        a = ATM_MODELS[self.model]['a']
        b = ATM_MODELS[self.model]['b']
        c = ATM_MODELS[self.model]['c']
        layers = ATM_MODELS[self.model]['h']
        y = np.where(h < layers[0], a[0] + b[0] * np.exp(-1 * h / c[0]), a[1] + b[1] * np.exp(-1 * h / c[1]))
        y = np.where(h < layers[1], y, a[2] + b[2] * np.exp(-1 * h / c[2]))
        y = np.where(h < layers[2], y, a[3] + b[3] * np.exp(-1 * h / c[3]))
        y = np.where(h < layers[3], y, a[4] - b[4] * h / c[4])
        y = np.where(h < H_MAX, y, 0)
        return y / np.cos(zenith)

    def get_vertical_height(self, zenith, xmax):
        """
        returns the (vertical) height above see level [in meters]
        as a function of zenith angle and Xmax [in g/cm^2]
        """
        return self._get_vertical_height(zenith, xmax * 1e4)

    def _get_vertical_height(self, zenith, x):
        """get vertical height for given zenith and atmospheric depth"""
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            args = self.__get_arguments(mask_numeric, zenith, x)
            tmp[mask_numeric] = self._get_vertical_height_numeric(*args)  # pylint: disable=E1120
        if np.sum(mask_taylor):
            raise NotImplementedError("""Taylor is not implemented (yet)""")
            # args = self.__get_arguments(mask_taylor, zenith, x)
            # tmp[mask_taylor] = self._get_vertical_height_numeric_taylor(*args)
        if np.sum(mask_flat):
            args = self.__get_arguments(mask_flat, zenith, x)
            tmp[mask_flat] = self._get_vertical_height_flat(*args)  # pylint: disable=E1120
        return tmp

    @staticmethod
    def _get_vertical_height_flat(zenith, x):
        """ Height above ground for given distance and zenith angle"""
        return overburden2height(x * np.cos(zenith) / 1E4)

    def get_density(self, zenith, xmax):
        """ Returns the atmospheric density as a function of zenith angle
            and shower maximum Xmax (in g/cm^2)
        """
        return self._get_density(zenith, xmax * 1e4)

    def _get_density(self, zenith, xmax):
        """ Returns the atmospheric density as a function of zenith angle
            and shower maximum Xmax
        """
        h = self._get_vertical_height(zenith, xmax)
        return density(h, model=self.model)

    def _get_density4(self, d, zenith):
        h = distance2height(d, zenith)
        return density(h, model=self.model)

    def get_distance_xmax(self, zenith, xmax, observation_level=1564.):
        """ Returns the atmospheric distance in [g/cm^2]
            for given zenith angle, xmax [g/cm^2] and observation level.
        """
        dxmax = self._get_distance_xmax(zenith, xmax * 1e4, observation_level=observation_level)
        return dxmax * 1e-4

    def _get_distance_xmax(self, zenith, xmax, observation_level=1564.):
        return self._get_atmosphere(zenith, h_low=observation_level) - xmax

    def get_distance_xmax_geometric(self, zenith, xmax, observation_level=1564.):
        """ Returns the atmospheric distance in [g/cm^2]
            for given zenith angle, xmax [g/cm^2] and observation level.
            This use the geometric approximation.
        """
        return self._get_distance_xmax_geometric(zenith, xmax * 1e4, observation_level=observation_level)

    def _get_distance_xmax_geometric(self, zenith, xmax, observation_level=1564.):
        h = self._get_vertical_height(zenith, xmax)
        return height2distance(h, zenith, observation_level)
