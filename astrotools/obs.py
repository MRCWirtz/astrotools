"""
Cosmic ray observables
"""
import numpy as np

import astrotools.coord as coord


def two_pt_auto(v, bins=180, **kwargs):
    """
    Angular two-point auto correlation for a set of directions v.
    WARNING: Due to the vectorized calculation this function
    does not work for large numbers of events.

    :param v: directions, (3 x n) matrix with the rows holding x,y,z
    :param bins: number of angular bins or np.ndarray with bin in radians
    :param kwargs: additional named arguments

                   - weights : weights for each event (optional)
                   - cumulative : make cumulative (default=True)
                   - normalized : normalize to 1 (default=False)
    """
    if isinstance(bins, int):
        bins = np.linspace(0, np.pi, bins+1)
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


def energy_energy_correlation(vec, energy, vec_roi, alpha_max=0.25, nbins=10, **kwargs):
    """
    Calculates the Energy-Energy-Correlation (EEC) of a given dataset for a given ROI.

    :param vec: arrival directions of CR events shape=(3, n)
    :param energy: energies of CR events in [EeV]
    :param vec_roi: position of ROI center shape=(3,)
    :param alpha_max: radial extend of ROI in radians
    :param nbins: number of angular bins in ROI
    :param kwargs: Additional keyword arguments
                   - bin_type: indicates if binning is linear in alpha ('lin')
                               or with equal area covered per bin ('area')
                   - e_ref: indicates if the 'mean' or the 'median' is taken for the average energy
                   - ref_mode: indicates if 'e_ref' is taken within 'bin' or the whole 'roi'
                   - verbose: if False, dont print warnings
    :return: alpha_bins: angular binning
    :return: omega_mean: mean values of EEC
    :return: ncr_bin: average number of CR in each angular bin
    """
    assert len(vec_roi) == 3
    bins = np.arange(nbins+1).astype(np.float)
    if kwargs.get("bin_type", 'area') == 'lin':
        alpha_bins = alpha_max * bins / nbins
    else:
        alpha_bins = 2 * np.arcsin(np.sqrt(bins/nbins) * np.sin(alpha_max/2))

    # angular distances to Center of each ROI
    dist_to_rois = coord.angle(vec_roi, vec, each2each=True)

    # instantiate output arrays
    omega_ij_list = [np.array([])] * nbins
    omega = np.zeros(nbins)
    ncr_roi_bin = np.zeros(nbins)

    # select CRs inside ROI
    mask_in_roi = dist_to_rois[0] < alpha_max     # type: np.ndarray
    ncr = int(vec[:, mask_in_roi].shape[1])
    e_cr = energy[mask_in_roi]

    # indices of angular bin for each CR
    alpha_cr = dist_to_rois[0, mask_in_roi]
    idx_cr = np.digitize(alpha_cr, alpha_bins) - 1

    # energy reference mean roi-wise
    if kwargs.get("ref_mode", 'bin') == 'roi':
        e_ref_roi = getattr(np, kwargs.get("e_ref", 'mean'))(e_cr)
        e_ref = np.repeat(e_ref_roi, nbins)
    # energy reference mean bin-wise
    else:
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
        ncr_roi_bin[i] = len(alpha_cr[idx_cr == i])
        mask_bin = (np.repeat(idx_cr, ncr).reshape((ncr, ncr)) == i) * (np.identity(ncr) == 0)
        omega_ij_list[i] = np.append(omega_ij_list[i], omega_ij[mask_bin])

        if (len(omega_ij_list[i]) == 0) and (kwargs.get('verbose', True)):
            print('Warning: Binning in dalpha is too small; no cosmic rays in bin %i.' % i)
            continue
        omega[i] = np.mean(omega_ij_list[i])

    return omega, alpha_bins, ncr_roi_bin
