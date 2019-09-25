"""
Collection of useful statistic related functionalities.
"""
import numpy as np


def mid(x):
    """
    Midpoints of a given array

    :param x: array with dimension bigger 1
    :return: all the midpoints as numpy array (shape: x.size -1)
    """
    return (x[:-1] + x[1:]) / 2.


def mean_and_variance(y, weights=None):
    """
    Weighted mean and variance of array y and weights

    :param y: array for which to calculate mean and variance
    :param weights: optional weights for weighted mean and variance
    :return: mean, weights (both like y dimensions)
    """
    weights = weights if weights is not None else np.ones(y.shape)
    assert y.shape == weights.shape
    w_sum = sum(weights)
    if w_sum.sum() == 0:
        return np.nan, np.nan
    m = np.dot(y, weights) / w_sum
    v = np.dot((y - m)**2, weights) / w_sum
    return m, v


def quantile_1d(data, weights, quant):
    """
    Compute the weighted quantile of a 1D numpy array.
    from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py

    :param data: 1d-array for which to calculate mean and variance
    :param weights : 1d-array with weights for data (same shape of data)
    :param quant: quantile to compute, it must have a value between 0 and 1.
    :return quantiles.
    """
    # Check the data
    if not isinstance(data, np.matrix):
        data = np.asarray(data)
    if not isinstance(weights, np.matrix):
        weights = np.asarray(weights)
    nd = data.ndim
    if nd != 1:
        raise TypeError("data must be a one dimensional array")
    ndw = weights.ndim
    if ndw != 1:
        raise TypeError("weights must be a one dimensional array")
    if data.shape != weights.shape:
        raise TypeError("the length of data and weights must be the same")
    if (quant > 1.) or (quant < 0.):
        raise ValueError("quantile must have a value between 0. and 1.")
    # Sort the data
    ind_sorted = np.argsort(data)
    sorted_data = data[ind_sorted]
    sorted_weights = weights[ind_sorted]
    # Compute the auxiliary arrays
    sn = np.cumsum(sorted_weights)
    assert np.sum(sn) > 0, "The sum of the weights must not be zero!"
    pn = (sn - 0.5 * sorted_weights) / np.sum(sorted_weights)
    # Get the value of the weighted median
    # noinspection PyTypeChecker
    return np.interp(quant, pn, sorted_data)


def quantile(data, weights, quant):  # pylint: disable=R1710
    """
    Weighted quantile of an array with respect to the last axis.
    from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py

    :param data: ndarray for which to calculate weighted quantile
    :param weights: ndarray with weights for data, it must have the same size
                    of the last axis of data.
    :param quant: quantile to compute, it must have a value between 0 and 1.
    :return: weighted quantiles with respect to last axis
    """
    # TODO: Allow to specify the axis
    nd = data.ndim
    assert nd > 0, "Data must have at least one dimension!"
    if nd == 1:
        return quantile_1d(data, weights, quant)
    n = data.shape
    assert n[-1] == weights.size, "Weights must have same size than last axis of data!"
    imr = data.reshape((-1, n[-1]))
    result = np.apply_along_axis(quantile_1d, -1, imr, weights, quant)
    return result.reshape(n[:-1])


def median(data, weights):
    """
    Weighted median of an array with respect to the last axis.
    Alias for quantile(data, weights, 0.5).
    from https://github.com/nudomarinero/wquantiles/blob/master/weighted.py

    :param data: ndarray for which to calculate the weighted median
    :param weights: ndarray with weights for data, it must have the same size
                    of the last axis of data.
    """
    return quantile(data, weights, 0.5)


def binned_mean_and_variance(x, y, bins, weights=None):
    """
    Calculates the mean and variance of y in the bins of x. This is effectively a ROOT.TProfile.

    :param x: data values that are used for binning (e.g. energies)
    :param y: data values that should be avaraged
    :param bins: bin borders
    return: <y>_i, sigma(y)_i : mean and variance of y in bins of x
    """
    dig = np.digitize(x, bins)
    n = len(bins) - 1
    my, vy = np.zeros(n), np.zeros(n)

    for i in range(n):
        idx = (dig == i+1)

        if not idx.any():  # check for empty bin
            my[i] = np.nan
            vy[i] = np.nan
            continue

        if weights is None:
            my[i] = np.mean(y[idx])
            vy[i] = np.std(y[idx])**2
        else:
            my[i], vy[i] = mean_and_variance(y[idx], weights[idx])

    return my, vy


def sym_interval_around(x, xm, alpha=0.32):
    """
    In a distribution represented by a set of samples, find the interval that contains (1-alpha)/2 to each the left and
    right of xm. If xm is too marginal to allow both sides to contain (1-alpha)/2, add the remaining fraction to the
    other side.

    :param x: data values in the distribution
    :param xm: symmetric center value for which to find the interval
    :param alpha: fraction that will be outside of the interval (default 0.32, corresponds to 68 percent quantile)
    :return: interval (lower, upper) which contains 1-alpha symmetric around xm
    """
    assert (alpha > 0) & (alpha < 1)
    xt = x.copy()
    xt.sort()
    i = xt.searchsorted(xm)  # index of central value
    n = len(x)  # number of samples
    ns = int((1 - alpha) * n)  # number of samples corresponding to 1-alpha

    i0 = i - ns/2  # index of lower and upper bound of interval
    i1 = i + ns/2

    # if central value does not allow for (1-alpha)/2 on left side, add to right
    if i0 < 0:
        i0 = 0
        i1 -= i0
    # if central value does not allow for (1-alpha)/2 on right side, add to left
    if i1 >= n:
        i1 = n-1
        i0 -= i1-n+1

    return xt[int(i0)], xt[int(i1)]
