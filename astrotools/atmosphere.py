import numpy as np
from scipy import integrate, interpolate, optimize


r_e = 6.371 * 1e6  # radius of Earth
h_max = 112829.2  # height above sea level where the mass overburden vanishes

"""
Atmospheric density models as used in CORSIKA.
The parameters are documented in the CORSIKA manual
The parameters for the Auger atmospheres are documented in detail in GAP2011-133
The May and October atmospheres describe the annual average best.
parameters
    a in g/cm^2
    b in g/cm^2
    c in cm
    h in km, layers
"""
default_model = 17
atm_models = {
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
    r = r_e + observation_level
    x = d * np.sin(zenith)
    y = d * np.cos(zenith) + r
    h = (x**2 + y**2)**0.5 - r
    return h


def height2distance(h, zenith, observation_level=0):
    """Distance for given height above ground and zenith angle"""
    r = r_e + observation_level
    return (h**2 + 2 * r * h + r**2 * np.cos(zenith)**2)**0.5 - r * np.cos(zenith)


def height2overburden(h, model=default_model):
    """Amount of atmosphere above given height.
    Args:
        h: height above sea level in meter
    Returns:
        atmospheric overburden in g/cm^2
    """
    a = atm_models[model]['a']
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']
    h = np.array(h)
    x = np.zeros_like(h)
    i = layers.searchsorted(h) - 1
    i = np.clip(i, 0, None)  # use layer 0 for negative heights
    x = np.where(i < 4,
                 a[i] + b[i] * np.exp(-h / c[i]),
                 a[4] - b[4] * h / c[4])
    x = np.where(h > h_max, 0, x)
    return x * 1E-4


def overburden2height(x, model=default_model):
    """Height for given overburden.
    Args:
        x: atmospheric overburden in g/cm^2
    Returns:
        height above sea level in meter
    """
    a = atm_models[model]['a']
    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']
    xlayers = height2overburden(layers, model=model)
    x = np.array(x)
    h = np.zeros_like(x)
    i = xlayers.size - np.searchsorted(xlayers[::-1], x) - 1
    i = np.clip(i, 0, None)
    h = np.where(i < 4,
                 -c[i] * np.log((x * 1E4 - a[i]) / b[i]),
                 -c[4] * (x * 1E4 - a[4]) / b[4])
    h = np.where(x <= 0, h_max, h)
    return h


def density(h, model=default_model):
    """Atmospheric density at given height
    Args:
        h: height above sea level in m
    Returns:
        atmospheric overburden in g/m^3
    """
    h = np.array(h)
    density = np.zeros_like(h)

    if model == 'barometric':  # barometric formula
        R = 8.31432  # universal gas constant for air: 8.31432 N m/(mol K)
        g0 = 9.80665  # gravitational acceleration (9.80665 m/s2)
        M = 0.0289644  # molar mass of Earth's air (0.0289644 kg/mol)
        rb = [1.2250, 0.36391, 0.08803, 0.01322, 0.00143, 0.00086, 0.000064]
        Tb = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65]
        Lb = [-0.0065, 0, 0.001, 0.0028, 0, -0.0028, -0.002]
        hb = [0, 11000, 20000, 32000, 47000, 51000, 71000]

        def rho1(h, i):  # for Lb == 0
            return rb[i] * np.exp(-g0 * M * (h - hb[i]) / (R * Tb[i]))

        def rho2(h, i):  # for Lb != 0
            return rb[i] * (Tb[i] / (Tb[i] + Lb[i] * (h - hb[i])))**(1 + (g0 * M) / (R * Lb[i]))

        i = np.searchsorted(hb, h) - 1
        density = np.where(Lb[i] == 0, rho1(h, i), rho2(h, i))
        density = np.where(h > 86000, 0, density)
        return density * 1e3

    b = atm_models[model]['b']
    c = atm_models[model]['c']
    layers = atm_models[model]['h']
    i = np.searchsorted(layers, h) - 1
    density = np.where(i < 4, np.exp(-h / c[i]), 1) * b[i] / c[i]
    return density


def refractive_index(h, n0=1.000292, model=default_model):
    """Refractive index at given height.

    Args:
        h (array): height above sea level in [m]
        n0 (float, optional): refractive index at sea level
        model (int, optional): atmospheric model

    Returns:
        array: refractive index at given height
    """
    return 1 + (n0 - 1) * density(h, model) / density(0, model)


class Atmosphere():
    """Atmosphere class from radiotools.
    Could use some refactoring.
    """

    def __init__(self, model=17, n_taylor=5, curved=True, zenith_numeric=np.deg2rad(83)):
        print('Loading model %i' % model)
        self.model = model
        self.curved = curved
        self.n_taylor = n_taylor
        self.__zenith_numeric = zenith_numeric
        self.b = atm_models[model]['b']
        self.c = atm_models[model]['c']
        self.h = atm_models[model]['h']
        self.num_zenith = 101
        if curved:
            print('Calculating constants for curved atmosphere')
            self.__zenith_numeric = 0
            self.d = np.zeros(self.num_zenith)
            self.a = self.__calculate_a()
            zenith = np.arccos(np.linspace(0, 1, self.num_zenith))
            mask = zenith < np.deg2rad(83)
            self.a_funcs = []
            for i in range(5):
                self.a_funcs.append(interpolate.interp1d(zenith[mask], self.a[:, i][mask], kind='cubic'))
            print('Done')

    def __calculate_a(self,):
        b = self.b
        c = self.c
        h = self.h
        a = np.zeros((self.num_zenith, 5))
        zenith = np.arccos(np.linspace(0, 1, self.num_zenith))
        for i, z in enumerate(zenith):
            print("zenith %.02f" % np.rad2deg(z))
            a[i, 0] = self._get_atmosphere_numeric([z], h_low=h[0]) - b[0]                        * self._get_dldh(h[0], z, 0)
            a[i, 1] = self._get_atmosphere_numeric([z], h_low=h[1]) - b[1] * np.exp(-h[1] / c[1]) * self._get_dldh(h[1], z, 1)
            a[i, 2] = self._get_atmosphere_numeric([z], h_low=h[2]) - b[2] * np.exp(-h[2] / c[2]) * self._get_dldh(h[2], z, 2)
            a[i, 3] = self._get_atmosphere_numeric([z], h_low=h[3]) - b[3] * np.exp(-h[3] / c[3]) * self._get_dldh(h[3], z, 3)
            a[i, 4] = self._get_atmosphere_numeric([z], h_low=h[4]) + b[4] * h[4] / c[4]          * self._get_dldh(h[4], z, 4)
        return a

    def _get_dldh(self, h, zenith, iH):
        if iH < 4:
            c = self.c[iH]
            st = np.sin(zenith)
            ct = np.cos(zenith)
            dldh = np.ones_like(zenith) / ct
            if self.n_taylor >= 1:
                dldh += -(st**2 / ct**3 * (c + h) / r_e)
            if self.n_taylor >= 2:
                dldh += 1.5 * st**2 * (2 * c**2 + 2 * c * h + h**2) / (r_e**2 * ct**5)
            if self.n_taylor >= 3:
                t1 = 6 * c**3 + 6 * c**2 * h + 3 * c * h**2 + h**3
                dldh += st**2 / (2 * r_e**3 * ct**7) * (ct**2 - 5) * t1
            if self.n_taylor >= 4:
                t1 = 24 * c**4 + 24 * c**3 * h + 12 * c**2 * h**2 + 4 * c * h**3 + h**4
                dldh += -1. * st**2 * 5. / (8. * r_e**4 * ct**9) * (3 * ct**2 - 7) * t1
            if self.n_taylor >= 5:
                t1 = 120 * c**5 + 120 * c**4 * h + 60 * c**3 * h**2 + 20 * c**2 * h**3 + 5 * c * h**4 + h**5
                dldh += st**2 * (ct**4 - 14. * ct**2 + 21.) * (-3. / 8.) / (r_e**5 * ct**11) * t1
        elif(iH == 4):
            c = self.c[iH]
            st = np.sin(zenith)
            ct = np.cos(zenith)
            dldh = np.ones_like(zenith) / ct
            if self.n_taylor >= 1:
                dldh += (-0.5 * st**2 / ct**3 * h / r_e)
            if self.n_taylor >= 2:
                dldh += 0.5 * st**2 / ct**5 * (h / r_e)**2
            if self.n_taylor >= 3:
                dldh += 1. / 8. * (st**2 * (ct**2 - 5) * h**3) / (r_e**3 * ct**7)
            if self.n_taylor >= 4:
                dldh += -1. / 8. * st**2 * (3 * ct**2 - 7) * (h / r_e)**4 / ct**9
            if self.n_taylor >= 5:
                dldh += -1. / 16. * st**2 * (ct**4 - 14 * ct**2 + 21) * (h / r_e)**5 / ct**11
        else:
            print("ERROR, height index our of bounds")

        return dldh

    def __get_method_mask(self, zenith):
        if not self.curved:
            return np.ones_like((3, zenith), dtype=np.bool)
        mask_flat = np.zeros_like(zenith, dtype=np.bool)
        mask_taylor = zenith < self.__zenith_numeric
        mask_numeric = zenith >= self.__zenith_numeric
        return mask_flat, mask_taylor, mask_numeric

    def __get_height_masks(self, hh):
        mask0 = (hh < atm_models[self.model]['h'][0])
        mask1 = (hh >= atm_models[self.model]['h'][0]) & (hh < atm_models[self.model]['h'][1])
        mask2 = (hh >= atm_models[self.model]['h'][1]) & (hh < atm_models[self.model]['h'][2])
        mask3 = (hh >= atm_models[self.model]['h'][2]) & (hh < atm_models[self.model]['h'][3])
        mask4 = (hh >= atm_models[self.model]['h'][3]) & (hh < h_max)
        mask5 = hh >= h_max
        return np.array([mask0, mask1, mask2, mask3, mask4, mask5])

    def __get_X_masks(self, X, zenith):
        mask0 = X > self._get_atmosphere(zenith, atm_models[self.model]['h'][0])
        mask1 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][0])) & \
                (X > self._get_atmosphere(zenith, atm_models[self.model]['h'][1]))
        mask2 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][1])) & \
                (X > self._get_atmosphere(zenith, atm_models[self.model]['h'][2]))
        mask3 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][2])) & \
                (X > self._get_atmosphere(zenith, atm_models[self.model]['h'][3]))
        mask4 = (X <= self._get_atmosphere(zenith, atm_models[self.model]['h'][3])) & \
                (X > self._get_atmosphere(zenith, h_max))
        mask5 = X <= 0
        return np.array([mask0, mask1, mask2, mask3, mask4, mask5])

    def __get_arguments(self, mask, *args):
        tmp = []
        ones = np.ones(np.array(mask).size)
        for a in args:
            if np.shape(a) == ():
                tmp.append(a * ones)
            else:
                tmp.append(a[mask])
        return tmp

    def get_atmosphere(self, zenith, h_low=0., h_up=np.infty):
        """ returns the atmosphere for an air shower with given zenith angle (in g/cm^2) """
        return self._get_atmosphere(zenith, h_low=h_low, h_up=h_up) * 1e-4

    def _get_atmosphere(self, zenith, h_low=0., h_up=np.infty):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        mask_finite = np.array((h_up * np.ones_like(zenith)) < h_max)
        is_mask_finite = np.sum(mask_finite)
        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            tmp[mask_numeric] = self._get_atmosphere_numeric(*self.__get_arguments(mask_numeric, zenith, h_low, h_up))
        if np.sum(mask_taylor):
            tmp[mask_taylor] = self._get_atmosphere_taylor(*self.__get_arguments(mask_taylor, zenith, h_low))
            if(is_mask_finite):
                mask_tmp = np.squeeze(mask_finite[mask_taylor])
                tmp2 = self._get_atmosphere_taylor(*self.__get_arguments(mask_taylor, zenith, h_up))
                tmp[mask_tmp] = tmp[mask_tmp] - np.array(tmp2)
        if np.sum(mask_flat):
            tmp[mask_flat] = self._get_atmosphere_flat(*self.__get_arguments(mask_flat, zenith, h_low))
            if(is_mask_finite):
                mask_tmp = np.squeeze(mask_finite[mask_flat])
                tmp2 = self._get_atmosphere_flat(*self.__get_arguments(mask_flat, zenith, h_up))
                tmp[mask_tmp] = tmp[mask_tmp] - np.array(tmp2)
        return tmp

    def _get_atmosphere_taylor(self, zenith, h_low=0.):
        b = self.b
        c = self.c
        a = np.c_[[self.a_funcs[i](zenith) for i in range(5)]]

        masks = self.__get_height_masks(h_low)
        tmp = np.zeros_like(zenith)
        for iH, mask in enumerate(masks):
            if(np.sum(mask)):
                if(np.array(h_low).size == 1):
                    h = h_low
                else:
                    h = h_low[mask]
                if iH < 4:
                    dldh = self._get_dldh(h, zenith[mask], iH)
                    tmp[mask] = np.array([a[..., iH][mask] + b[iH] * np.exp(-1 * h / c[iH]) * dldh])
                elif iH == 4:
                    dldh = self._get_dldh(h, zenith[mask], iH)
                    tmp[mask] = np.array([a[..., iH][mask] - b[iH] * h / c[iH] * dldh])
                else:
                    tmp[mask] = np.zeros(np.sum(mask))
        return tmp

    def _get_atmosphere_numeric(self, zenith, h_low=0, h_up=np.infty):
        zenith = np.array(zenith)
        tmp = np.zeros_like(zenith)
        for i in range(len(tmp)):
            if(np.array(h_up).size == 1):
                t_h_up = h_up
            else:
                t_h_up = h_up[i]
            if(np.array(h_low).size == 1):
                t_h_low = h_low
            else:
                t_h_low = h_low[i]
            z = zenith[i]
            if t_h_up <= t_h_low:
                print("WARNING _get_atmosphere_numeric(): upper limit less than lower limit")
                return np.nan
            if t_h_up == np.infty:
                t_h_up = h_max
            b = t_h_up
            d_low = height2distance(t_h_low, z)
            d_up = height2distance(b, z)
            full_atm = integrate.quad(self._get_density4, d_low, d_up, args=(z,), limit=500)[0]
            tmp[i] = full_atm
        return tmp

    def _get_atmosphere_flat(self, zenith, h=0):
        a = atm_models[self.model]['a']
        b = atm_models[self.model]['b']
        c = atm_models[self.model]['c']
        layers = atm_models[self.model]['h']
        y = np.where(h < layers[0], a[0] + b[0] * np.exp(-1 * h / c[0]), a[1] + b[1] * np.exp(-1 * h / c[1]))
        y = np.where(h < layers[1], y, a[2] + b[2] * np.exp(-1 * h / c[2]))
        y = np.where(h < layers[2], y, a[3] + b[3] * np.exp(-1 * h / c[3]))
        y = np.where(h < layers[3], y, a[4] - b[4] * h / c[4])
        y = np.where(h < h_max, y, 0)
        return y / np.cos(zenith)

    def get_vertical_height(self, zenith, xmax):
        """ returns the (vertical) height above see level [in meters] as a function of zenith angle and Xmax [in g/cm^2] """
        return self._get_vertical_height(zenith, xmax * 1e4)

    def _get_vertical_height(self, zenith, X):
        mask_flat, mask_taylor, mask_numeric = self.__get_method_mask(zenith)
        tmp = np.zeros_like(zenith)
        if np.sum(mask_numeric):
            print("get vertical height numeric")
            tmp[mask_numeric] = self._get_vertical_height_numeric(*self.__get_arguments(mask_numeric, zenith, X))
        if np.sum(mask_taylor):
            tmp[mask_taylor] = self._get_vertical_height_numeric_taylor(*self.__get_arguments(mask_taylor, zenith, X))
        if np.sum(mask_flat):
            print("get vertical height flat")
            tmp[mask_flat] = self._get_vertical_height_flat(*self.__get_arguments(mask_flat, zenith, X))
        return tmp

    def _get_vertical_height_taylor(self, zenith, X):
        def get_zenith_a_indices(self, zeniths):
            n = self.num_zenith - 1
            cosz_bins = np.linspace(0, n, self.num_zenith, dtype=np.int)
            cosz = np.array(np.round(np.cos(zeniths) * n), dtype=np.int)
            tmp = np.squeeze([np.argwhere(t == cosz_bins) for t in cosz])
            return tmp
        tmp = self._get_vertical_height_taylor_wo_constants(zenith, X)
        masks = self.__get_X_masks(X, zenith)
        d = self.d[get_zenith_a_indices(zenith)]
        for iX, mask in enumerate(masks):
            if(np.sum(mask)):
                if iX < 4:
                    print(mask)
                    print(tmp[mask], len(tmp[mask]))
                    print(d[mask][..., iX])
                    tmp[mask] += d[mask][..., iX]
        return tmp

    def _get_vertical_height_taylor_wo_constants(self, zenith, X):
        b = self.b
        c = self.c
        ct = np.cos(zenith)
        T0 = self._get_atmosphere(zenith)
        masks = self.__get_X_masks(X, zenith)
        tmp = np.zeros_like(zenith)
        for iX, mask in enumerate(masks):
            if(np.sum(mask)):
                if iX < 4:
                    xx = X[mask] - T0[mask]
                    if self.n_taylor >= 1:
                        tmp[mask] = -c[iX] / b[iX] * ct[mask] * xx
                    if self.n_taylor >= 2:
                        tmp[mask] += -0.5 * c[iX] * (ct[mask]**2 * c[iX] - ct[mask]**2 * r_e - c[iX]) / (r_e * b[iX]**2) * xx**2
                    if self.n_taylor >= 3:
                        tmp[mask] += -1. / 6. * c[iX] * ct[mask] * (3 * ct[mask]**2 * c[iX]**2 - 4 * ct[mask]**2 * r_e * c[iX] + 2 * r_e**2 * ct[mask]**2 - 3 * c[iX]**2 + 4 * r_e * c[iX]) / (r_e**2 * b[iX]**3) * xx**3
                    if self.n_taylor >= 4:
                        tmp[mask] += -1. / (24. * r_e**3 * b[iX]**4) * c[iX] * (15 * ct[mask]**4 * c[iX]**3 - 25 * c[iX]**2 * r_e * ct[mask]**4 + 18 * c[iX] * r_e**2 * ct[mask]**4 - 6 * r_e**3 * ct[mask]**4 - 18 * c[iX]**3 * ct[mask]**2 + 29 * c[iX]**2 * r_e * ct[mask]**2 - 18 * c[iX] * r_e**2 * ct[mask]**2 + 3 * c[iX]**3 - 4 * c[iX]**2 * r_e) * xx**4
                    if self.n_taylor >= 5:
                        tmp[mask] += -1. / (120. * r_e**4 * b[iX]**5) * c[iX] * ct[mask] * (ct[mask]**4 * (105 * c[iX]**4 - 210 * c[iX]**3 * r_e + 190 * c[iX]**2 * r_e**2 - 96 * c[iX] * r_e**3 + 24 * r_e**4) + ct[mask]**2 * (-150 * c[iX]**4 + 288 * c[iX]**3 * r_e - 242 * c[iX]**2 * r_e**2 + 96 * c[iX] * r_e**3) + 45 * c[iX]**4 - 78 * r_e * c[iX]**3 + 52 * r_e**2 * c[iX]**2) * xx**5
                    if self.n_taylor >= 6:
                        tmp[mask] += -1. / (720. * r_e**5 * b[iX]**6) * c[iX] * (ct[mask]**6 * (945 * c[iX]**5 - 2205 * c[iX]**4 * r_e + 2380 * c[iX]**3 * r_e**2 - 1526 * c[iX]**2 * r_e**3 + 600 * c[iX] * r_e**4 - 120 * r_e**5) + ct[mask]**4 * (-1575 * c[iX]**5 + 3528 * c[iX]**4 * r_e - 3600 * c[iX]**3 * r_e**2 + 2074 * c[iX]**2 * r_e**3 - 600 * c[iX] * r_e**4) + ct[mask]**2 * (675 * c[iX]**5 - 1401 * c[iX]**4 * r_e - 1272 * c[iX]**3 * r_e**2 - 548 * c[iX]**2 * r_e**3) - 45 * c[iX]**5 + 78 * c[iX]**4 * r_e - 52 * c[iX]**3 * r_e**2) * xx**6
                elif iX == 4:
                    print("iX == 4", iX)
                    # numeric fall-back
                    tmp[mask] = self._get_vertical_height_numeric(zenith, X)
                else:
                    print("iX > 4", iX)
                    tmp[mask] = np.ones_like(mask) * h_max
        return tmp

    def _get_vertical_height_numeric(self, zenith, X):
        tmp = np.zeros_like(zenith)
        zenith = np.array(zenith)
        for i in range(len(tmp)):

            x0 = height2distance(self._get_vertical_height_flat(zenith[i], X[i]), zenith[i])

            def ftmp(d, zenith, xmax, observation_level=0):
                h = distance2height(d, zenith, observation_level=observation_level)
                h += observation_level
                tmp = self._get_atmosphere_numeric([zenith], h_low=h)
                dtmp = tmp - xmax
                return dtmp

            dxmax_geo = optimize.brentq(ftmp, -1e3, x0 + 1e4, xtol=1e-6,
                                        args=(zenith[i], X[i]))
            tmp[i] = distance2height(dxmax_geo, zenith[i])
        return tmp

    def _get_vertical_height_numeric_taylor(self, zenith, X):
        tmp = np.zeros_like(zenith)
        zenith = np.array(zenith)
        for i in range(len(tmp)):
            x0 = height2distance(self._get_vertical_height_flat(zenith[i], X[i]), zenith[i])

            def ftmp(d, zenith, xmax, observation_level=0):
                h = distance2height(d, zenith, observation_level=observation_level)
                h += observation_level
                tmp = self._get_atmosphere_taylor(np.array([zenith]), h_low=np.array([h]))
                dtmp = tmp - xmax
                return dtmp

            dxmax_geo = optimize.brentq(ftmp, -1e3, x0 + 1e4, xtol=1e-6,
                                        args=(zenith[i], X[i]))
            tmp[i] = distance2height(dxmax_geo, zenith[i])
        return tmp

    def _get_vertical_height_flat(self, zenith, X):
        """Height above ground for given distance and zenith angle"""
        return overburden2height(X * np.cos(zenith) / 1E4)

    def get_density(self, zenith, xmax):
        """ returns the atmospheric density as a function of zenith angle
        and shower maximum Xmax (in g/cm^2) """
        return self._get_density(zenith, xmax * 1e4)

    def _get_density(self, zenith, xmax):
        """ returns the atmospheric density as a function of zenith angle
        and shower maximum Xmax """
        h = self._get_vertical_height(zenith, xmax)
        rho = density(h, model=self.model)
        return rho

    def _get_density4(self, d, zenith):
        h = distance2height(d, zenith)
        return density(h, model=self.model)

    def get_distance_xmax(self, zenith, xmax, observation_level=1564.):
        """ input:
            - xmax in g/cm^2
            - zenith in radians
            output: distance to xmax in g/cm^2
        """
        dxmax = self._get_distance_xmax(zenith, xmax * 1e4, observation_level=observation_level)
        return dxmax * 1e-4

    def _get_distance_xmax(self, zenith, xmax, observation_level=1564.):
        return self._get_atmosphere(zenith, h_low=observation_level) - xmax

    def get_distance_xmax_geometric(self, zenith, xmax, observation_level=1564.):
        """ input:
            - xmax in g/cm^2
            - zenith in radians
            output: distance to xmax in m
        """
        return self._get_distance_xmax_geometric(zenith, xmax * 1e4, observation_level=observation_level)

    def _get_distance_xmax_geometric(self, zenith, xmax, observation_level=1564.):
        h = self._get_vertical_height(zenith, xmax)
        return height2distance(h, zenith, observation_level)
