# astrotools.stat
# Some useful statistic functions
import numpy as np


def mid(x):
    """
    Midpoints of a given array
    """
    return (x[:-1] + x[1:]) / 2.


def meanAndVariance(y, weights):
    """
    Weighted mean and variance
    """
    wSum = sum(weights)
    m = np.dot(y, weights) / wSum
    v = np.dot((y - m)**2, weights) / wSum
    return m, v


def binnedMean(x, y, bins, weights=None):
    """
    <y>_i : mean of y in bins of x
    """
    dig = np.digitize(x, bins)
    n = len(bins) - 1
    my = np.zeros(n)
    if weights is None:
        weights = np.ones(len(x))  # use weights=1 if none given

    for i in range(n):
        idx = (dig == i+1)
        try:
            my[i] = np.average(y[idx], weights=weights[idx])
        except ZeroDivisionError:
            my[i] = np.nan

    return my


def binnedMeanAndVariance(x, y, bins, weights=None):
    """
    <y>_i, sigma(y)_i : mean and variance of y in bins of x
    This is effectively a ROOT.TProfile
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
            my[i], vy[i] = meanAndVariance(y[idx], weights[idx])

    return my, vy


def symIntervalAround(x, xm, alpha):
    """
    In a distribution represented by a set of samples, find the interval
    that contains (1-alpha)/2 to each the left and right of xm.
    If xm is too marginal to allow both sides to contain (1-alpha)/2,
    add the remaining fraction to the other side.
    """
    xt = x.copy()
    xt.sort()
    i = xt.searchsorted(xm)  # index of central value
    n = len(x)  # number of samples
    ns = int((1 - alpha) * n)  # number of samples corresponding to 1-alpha

    i0 = i - ns/2  # index of lower and upper bound of interval
    i1 = i + ns/2

    # if central value doesn't allow for (1-alpha)/2 on left side, add to right
    if i0 < 0:
        i1 -= i0
        i0 = 0
    # if central value doesn't allow for (1-alpha)/2 on right side, add to left
    if i1 >= n:
        i0 -= i1-n+1
        i1 = n-1

    return xt[i0], xt[i1]
