# astrotools.stat
# Some useful static functions when using weights
import numpy as np


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
    if weights == None:
        weights = np.ones(len(x)) # use weights=1 if none given

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
    This is effectively a ROOT.Profile
    """
    dig = np.digitize(x, bins)
    n = len(bins) - 1
    my, vy = np.zeros(n), np.zeros(n)

    for i in range(n):
        idx = (dig == i+1)

        if not idx.any(): #check for empty bin
            my[i] = np.nan
            vy[i] = np.nan
            continue

        if weights == None:
            my[i] = np.mean(y[idx])
            vy[i] = np.std(y[idx])**2
        else:
            my[i], vy[i] = meanAndVariance(y[idx], weights[idx])

    return my, vy
