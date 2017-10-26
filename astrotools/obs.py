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
def thrust(v, weights=None, ntry=5000):
    """
    Thrust observable for an array (n x 3) of 3-momenta.
    Returns 3 values (thrust, thrust major, thrust minor)
    and the corresponding axes.

    :param v: 3-momenta, (3 x n) matrix with the columns holding x,y,z
    :param weights: (optional) weights for each event, e.g. 1/exposure
    :param ntry: number of samples for the brute force computation of thrust major
    :return: tuple consisting of the following values
    
             - thrust, thrust major, thrust minor
             - thrust axis, thrust major axis, thrust minor axis
    """
    # optional weights
    v = (v * weights) if weights is not None else v

    # thrust
    n1 = np.sum(v, axis=0)
    n1 /= np.linalg.norm(n1)
    t1 = np.sum(abs(np.dot(v, n1)))

    # thrust major, brute force calculation
    er, et, ep = coord.sph_unit_vectors(*coord.vec2ang(n1)).T
    alpha = np.linspace(0, np.pi, ntry)
    n2_try = np.outer(np.cos(alpha), et) + np.outer(np.sin(alpha), ep)
    t2_try = np.sum(abs(np.dot(v, n2_try.T)), axis=0)
    i = np.argmax(t2_try)
    n2 = n2_try[i]
    t2 = t2_try[i]

    # thrust minor
    n3 = np.cross(n1, n2)
    t3 = np.sum(abs(np.dot(v, n3)))

    # normalize
    sumP = np.sum(np.sum(v ** 2, axis=1) ** .5)
    T = np.array((t1, t2, t3)) / sumP
    N = np.array((n1, n2, n3))
    return T, N

 
def energy_energy_correlation(v, log10e, v_roi, alpha_max=0.25, mean_energy_mode='roi', e_mean_median='mean', nbins=10, bin_type='lin'):
    """
    Calculates the Energy-Energy-Correlation (EEC) of a given dataset, averaged over all region of interests. 

    :param v: arrival directions of CR events (x, y, z)
    :param log10e: energies of CR events in log10(E/eV)
    :param v_roi: positions of centers of ROIs (x, y, z)
    :param alpha_max: radial extend of ROI in radians 
    :param mean_energy_mode: indicates if the average energies are taken within one 'roi' or are 'global' including all ROI at once 
    :param e_mean_median: indicates if the 'mean' or the 'median' is taken for the average energy
    :param nbins: number of angular bins in ROI
    :param bin_type: indicates if binning is linear in alpha ('lin') or with equal area covered per bin ('area')
    :return: alpha_bins: angular binning 
    :return: omega_mean: mean values of EEC 
    :return: ncr_roi: average number of CR in each angular bin
    """
    energy = 10**(log10e - 18.)
    bins = np.arange(nbins+1).astype(np.float)
    if bin_type == 'lin':
        alpha_bins = alpha_max * bins / nbins
    elif bin_type == 'area':
        alpha_bins = 2 * np.arcsin(np.sqrt(bins/nbins) * np.sin(alpha_max/2))
    else:
        raise Exception("Value of variable 'bin_type' not understood!")

    # angular distances to Center of each ROI
    dist_to_rois = coord.angle(v_roi, v, each2each=True)

    # mean energy including all ROI:
    e_mean = np.zeros(nbins)
    if mean_energy_mode == 'global':
        energies_in_bins = [np.array([]) for i in range(nbins)]

        for roi in range(v_roi.shape[1]):
            # CRs inside ROI
            alpha_cr_roi = dist_to_rois[roi, dist_to_rois[roi] < alpha_max]
            e_cr_roi = energy[dist_to_rois[roi] < alpha_max]

            # indices of angular bin for each CR
            idx = np.digitize(alpha_cr_roi, alpha_bins) - 1

            for i in range(nbins):
                mask_bin = idx == i # type: np.ndarray
                energies_in_bins[i] = np.append(energies_in_bins[i], e_cr_roi[mask_bin])

        for i in range(nbins):
            if len(energies_in_bins[i]) > 0:
                    if e_mean_median == 'mean':
                        e_mean[i] = np.mean(energies_in_bins[i])
                    if e_mean_median == 'median':
                        e_mean[i] = np.median(energies_in_bins[i])



    # list of arrays containing all Omega_ij of all roi for each angular bin
    omega_ij = [np.array([]) for i in range(nbins)]
    # average number of CRs in each bin and each roi
    ncr_bin_roi = np.zeros((v_roi.shape[1], nbins))

    for roi in range(v_roi.shape[1]):
        # CRs inside ROI
        ncr_roi = int(v[:, dist_to_rois[roi] < alpha_max].shape[1])
        alpha_cr_roi = dist_to_rois[roi, dist_to_rois[roi] < alpha_max]
        e_cr_roi = energy[dist_to_rois[roi] < alpha_max]

        # indices of angular bin for each CR
        idx = np.digitize(alpha_cr_roi, alpha_bins) - 1

        # mean energy in each bin
        if mean_energy_mode == 'roi':
            for i in range(nbins):
                mask_bin = idx == i # type: np.ndarray
                if np.sum(mask_bin) > 0:
                    if e_mean_median == 'mean':
                        e_mean[i] = np.mean(e_cr_roi[mask_bin])
                    if e_mean_median == 'median':
                        e_mean[i] = np.median(e_cr_roi[mask_bin])


        # Omega_ij for each pair of CRs in ROI
        omega_matrix = (np.array([e_cr_roi]) - np.array([e_mean[idx]])) / np.array([e_cr_roi])
        omega_ij_roi = omega_matrix * omega_matrix.T

        # sort Omega_ij into respective angular bins
        for i in range(nbins):
            # save number of CRs in each bin
            ncr_bin_roi[roi, i] = len(alpha_cr_roi[idx == i])
            mask_idx_i = (np.repeat(idx, ncr_roi).reshape((ncr_roi, ncr_roi)) == i) * (np.identity(ncr_roi) == 0)
            omega_ij[i] = np.append(omega_ij[i], omega_ij_roi[mask_idx_i])


    # global mean omega per bin
    omega_global_mean = np.zeros(len(omega_ij))

    for i, om in enumerate(omega_ij):
        if len(om) == 0:
            print('Warning: Binning in dalpha is too small; no cosmic rays in bin %i.' % i)
            continue
        omega_global_mean[i] = np.mean(om)

    # average number of CRs in each bin for this map
    ncr_bin = np.mean(ncr_bin_roi, axis=0)

    return alpha_bins, omega_global_mean, ncr_bin
