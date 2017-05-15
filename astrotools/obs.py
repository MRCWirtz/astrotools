"""
Cosmic ray observables
"""
import numpy as np
import astrotools.coord as coord


def two_pt_auto(v, bins=np.arange(0, 181, 1), **kwargs):
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
    ac = np.bincount(dig, minlength=len(bins)+1, weights=w)
    ac = ac.astype(float)[1:-1]  # convert to float and remove overflow bins

    if kwargs.get("cumulative", True):
        ac = np.cumsum(ac)
    if kwargs.get("normalized", False):
        if w is not None:
            ac /= sum(w)
        else:
            ac /= (n**2 - n) / 2
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

    cc = np.bincount(dig, minlength=len(bins)+1, weights=w)
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
def thrust(P, weights=None, ntry=5000):
    """
    Thrust observable for an array (n x 3) of 3-momenta.
    Returns 3 values (thrust, thrust major, thrust minor)
    and the corresponding axes.

    :param P: 3-momenta, (n x 3) matrix with the columns holding x,y,z
    :param weights: (optional) weights for each event, e.g. 1/exposure
    :param ntry: number of samples for the brute force computation of thrust major
    :return: tuple consisting of the following values
    
             - thrust, thrust major, thrust minor
             - thrust axis, thrust major axis, thrust minor axis
    """
    # optional weights
    if weights is not None:
        Pw = (P.T * weights).T
    else:
        Pw = P

    # thrust
    n1 = np.sum(Pw, axis=0)
    n1 /= np.linalg.norm(n1)
    t1 = np.sum(abs(np.dot(Pw, n1)))

    # thrust major, brute force calculation
    er, et, ep = coord.sph_unit_vectors(*coord.vec2ang(n1)).T
    alpha = np.linspace(0, np.pi, ntry)
    n2_try = np.outer(np.cos(alpha), et) + np.outer(np.sin(alpha), ep)
    t2_try = np.sum(abs(np.dot(P, n2_try.T)), axis=0)
    i = np.argmax(t2_try)
    n2 = n2_try[i]
    t2 = t2_try[i]

    # thrust minor
    n3 = np.cross(n1, n2)
    t3 = np.sum(abs(np.dot(Pw, n3)))

    # normalize
    sumP = np.sum(np.sum(Pw**2, axis=1)**.5)
    T = np.array((t1, t2, t3)) / sumP
    N = np.array((n1, n2, n3))
    return T, N


def energy_energy_correlation(vec, log10e, vec_roi, alpha_max=0.25, dalpha=0.025):
    """
    Calculates the Energy-Energy-Correlation (EEC) of a given dataset, averaged over all region of interests. 

    :param vec: arrival directions of CR events (x, y, z)
    :param log10e: energies of CR events in log10(E/eV)
    :param vec_roi: positions of centers of ROIs (x, y, z)
    :param alpha_max: radial extend of ROI in radians 
    :param dalpha: radial width of angular bins in radians
    :return: alpha_bins: angular binning 
    :return: omega_mean: mean values of EEC 
    """
    energy = 10**(log10e - 18.)
    alpha_bins = np.arange(0, alpha_max, dalpha)
    n_bins = len(alpha_bins)

    # angular distances to Center of ROI
    dist_to_rois = coord.angle(vec_roi, vec, each2each=True)

    # list of arrays containing all Omega_ij for each angular bin
    omega = [np.array([]) for i in range(n_bins)]

    for roi in range(len(vec_roi[0])):
        # CRs inside ROI
        ncr_roi = vec[:, dist_to_rois[roi] < alpha_max].shape[1]
        alpha_cr_roi = dist_to_rois[roi, dist_to_rois[roi] < alpha_max]
        e_cr_roi = energy[dist_to_rois[roi] < alpha_max]

        # indices of angular bin for each CR
        idx = np.digitize(alpha_cr_roi, alpha_bins) - 1

        # mean energy per dalpha
        e_mean = np.zeros(n_bins)
        for i, bin_i in enumerate(alpha_bins):
            mask_bin = (alpha_cr_roi >= bin_i) * (alpha_cr_roi < bin_i + dalpha)  # type: np.ndarray
            if np.sum(mask_bin) > 0:
                e_mean[i] = np.mean(e_cr_roi[mask_bin])

        # Omega_ij for each pair of CRs in ROI
        Omega_matrix = (np.array([e_cr_roi]) - np.array([e_mean[idx]])) / np.array([e_cr_roi])
        Omega_ij = Omega_matrix * Omega_matrix.T

        # sort Omega_ij into respective angular bins
        for i, bin_i in enumerate(alpha_bins):
            mask_idx_i = (np.repeat(idx, ncr_roi).reshape((ncr_roi, ncr_roi)) == i) * (np.identity(ncr_roi) == 0)
            omega[i] = np.append(omega[i], Omega_ij[mask_idx_i])

    # mean omega per dalpha
    omega_mean = np.zeros(len(omega))
    for i, om in enumerate(omega):
        if len(om) == 0:
            print('Warning: Binning in dalpha is too small; no cosmic rays in bin %i.' % i)
            continue
        omega_mean[i] = np.mean(om)
    return alpha_bins, omega_mean
