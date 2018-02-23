"""
Cosmic ray observables
"""
import numpy as np

import astrotools.coord as coord


def two_pt_auto(v, bins=np.linspace(0, np.pi, 181), **kwargs):
    """
    Angular two-point auto correlation for a set of directions v.
    WARNING: Due to the vectorized calculation this function
    does not work for large numbers of events.

    :param v: directions, (3 x n) matrix with the rows holding x,y,z
    :param bins: angular bins in degrees
    :param kwargs: additional named arguments

                   - weights : weights for each event (optional)
                   - cumulative : make cumulative (default=True)
                   - normalized : normalize to 1 (default=False)
    """
    n = np.shape(v)[1]
    idx = np.triu_indices(n, 1)  # upper triangle indices without diagonal
    ang = coord.angle(v, v, each2each=True)[idx]

    # optional weights
    w = kwargs.get('weights', None)
    if w is not None:
        w = np.outer(w, w)[idx]

    dig = np.digitize(ang, bins)
    ac = np.bincount(dig, minlength=len(bins) + 1, weights=w)
    ac = ac.astype(float)[1:-1]  # convert to float and remove overflow bins

    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
    if kwargs.get("normalized", False):
        if w is not None:
            ac /= sum(w)
        else:
            ac /= (n ** 2 - n) / 2
    return ac


def two_pt_cross(v1, v2, bins=np.arange(0, 181, 1), **kwargs):
    """
    Angular two-point cross correlation for two sets of directions v1, v2.

    :param v1: directions, (3 x n1) matrix with the rows holding x,y,z
    :param v2: directions, (3 x n2) matrix with the rows holding x,y,z
    :param bins: angular bins in degrees
    :param kwargs: additional named arguments

                   - weights1, weights2: weights for each event (optional)
                   - cumulative: make cumulative (default=True)
                   - normalized: normalize to 1 (default=False)
    """
    ang = coord.angle(v1, v2, each2each=True).flatten()
    dig = np.digitize(ang, bins)

    # optional weights
    w1 = kwargs.get('weights1', None)
    w2 = kwargs.get('weights2', None)
    if (w1 is not None) and (w2 is not None):
        w = np.outer(w1, w2).flatten()
    else:
        w = None

    cc = np.bincount(dig, minlength=len(bins) + 1, weights=w)
    cc = cc.astype(float)[1:-1]

    if kwargs.get("cumulative", True):
        cc = np.cumsum(cc)
    if kwargs.get("normalized", False):
        if w is not None:
            cc /= sum(w)
        else:
            n1 = np.shape(v1)[1]
            n2 = np.shape(v2)[1]
            cc /= n1 * n2
    return cc


# noinspection PyTypeChecker
def thrust(p, weights=None, ntry=1000):
    """
    Thrust observable for an array (n x 3) of 3-momenta.
    Returns 3 values (thrust, thrust major, thrust minor)
    and the corresponding axes.

    :param p: 3-momenta, (3 x n) matrix with the columns holding px, py, pz
    :param weights: (optional) weights for each event, e.g. 1/exposure (1 x n)
    :param ntry: number of samples for the brute force computation of thrust major
    :return: tuple consisting of the following values

             - thrust, thrust major, thrust minor
             - thrust axis, thrust major axis, thrust minor axis
    """
    # optional weights
    p = (p * weights) if weights is not None else p

    # thrust
    n1 = np.sum(p, axis=1)
    n1 /= np.linalg.norm(n1)
    t1 = np.sum(abs(np.dot(n1, p)))

    # thrust major, brute force calculation
    _, et, ep = coord.sph_unit_vectors(*coord.vec2ang(n1)).T
    alpha = np.linspace(0, np.pi, ntry)
    n2_try = np.outer(np.cos(alpha), et) + np.outer(np.sin(alpha), ep)
    t2_try = np.sum(abs(np.dot(n2_try, p)), axis=1)
    i = np.argmax(t2_try)
    n2 = n2_try[i]
    t2 = t2_try[i]

    # thrust minor
    n3 = np.cross(n1, n2)
    t3 = np.sum(abs(np.dot(n3, p)))

    # normalize
    sum_p = np.sum(np.sum(p ** 2, axis=0) ** .5)
    t = np.array((t1, t2, t3)) / sum_p
    n = np.array((n1, n2, n3))
    return t, n


def energy_energy_correlation(vec, log10e, vec_roi, alpha_max=0.25, nbins=10, **kwargs):
    """
    Calculates the Energy-Energy-Correlation (EEC) of a given dataset for given ROIs.

    :param vec: arrival directions of CR events (x, y, z)
    :param log10e: energies of CR events in log10(E/eV)
    :param vec_roi: positions of centers of ROIs (x, y, z)
    :param alpha_max: radial extend of ROI in radians
    :param nbins: number of angular bins in ROI
    :param kwargs: Additional keyword arguments
                   - bin_type: indicates if binning is linear in alpha ('lin')
                               or with equal area covered per bin ('area')
                   - e_ref: indicates if the 'mean' or the 'median' is taken for the average energy
    :return: alpha_bins: angular binning
    :return: omega_mean: mean values of EEC
    :return: ncr_bin: average number of CR in each angular bin
    """
    # energy = 10**(log10e - 18.)
    energy = log10e
    vec_roi = np.reshape(vec_roi, (3, -1))
    nroi = vec_roi.shape[1]
    bins = np.arange(nbins+1).astype(np.float)
    if kwargs.get("bin_type", 'area') == 'lin':
        alpha_bins = alpha_max * bins / nbins
    else:
        alpha_bins = 2 * np.arcsin(np.sqrt(bins/nbins) * np.sin(alpha_max/2))

    # angular distances to Center of each ROI
    dist_to_rois = coord.angle(vec_roi, vec, each2each=True)

    # calculate eec for each roi and each bin
    omega_ij_list = [[np.array([])] * nbins] * nroi
    omega = np.zeros((nroi, nbins))
    ncr_roi_bin = np.zeros((nroi, nbins))

    for roi in range(nroi):
        # CRs inside ROI
        mask_in_roi = dist_to_rois[roi] < alpha_max     # type: np.ndarray
        ncr = int(vec[:, mask_in_roi].shape[1])
        e_cr = energy[mask_in_roi]

        # indices of angular bin for each CR
        alpha_cr = dist_to_rois[roi, mask_in_roi]
        idx_cr = np.digitize(alpha_cr, alpha_bins) - 1

        # mean energy in each bin
        e_ref = np.zeros(nbins)
        for i in range(nbins):
            mask_bin = idx_cr == i  # type: np.ndarray
            if np.sum(mask_bin) > 0:
                e_ref[i] = getattr(np, kwargs.get("e_ref", 'mean'))(e_cr[mask_bin])

        # Omega_ij for each pair of CRs in whole ROI
        omega_matrix = (np.array([e_cr]) - np.array([e_ref[idx_cr]])) / np.array([e_cr])
        omega_ij = omega_matrix * omega_matrix.T

        # sort Omega_ij into respective angular bins
        for i in range(nbins):
            ncr_roi_bin[roi, i] = len(alpha_cr[idx_cr == i])
            mask_bin = (np.repeat(idx_cr, ncr).reshape((ncr, ncr)) == i) * (np.identity(ncr) == 0)
            omega_ij_list[roi][i] = np.append(omega_ij_list[roi][i], omega_ij[mask_bin])

            if len(omega_ij_list[roi][i]) == 0:
                print('Warning: Binning in dalpha is too small; no cosmic rays in bin %i.' % i)
                continue
            omega[roi, i] = np.mean(omega_ij_list[roi][i])

    return omega, alpha_bins, ncr_roi_bin
