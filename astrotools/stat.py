# astrotools.stat
# Some useful static functions when using weights

import numpy


def meanAndVariance(y, weights):
    """
    Weighted mean and variance
    """
    wSum = sum(weights)
    m = numpy.dot(y, weights) / wSum
    v = numpy.dot((y - m)**2, weights) / wSum
    return m, v

def binnedMean(x, y, bins, weights=None):
    """
    <y>_i : mean of y in bins of x
    """
    dig = numpy.digitize(x, bins)
    n = len(bins) - 1
    my = numpy.zeros(n)
    if weights == None:
        weights = numpy.ones(len(x)) # use weights=1 if none given

    for i in range(n):
        idx = (dig == i+1)
        try:
            my[i] = numpy.average(y[idx], weights=weights[idx])
        except RuntimeWarning: # catch errors on empty bins
            my[i] = numpy.nan

    return my

def binnedMeanAndVariance(x, y, bins, weights=None):
    """
    <y>_i, sigma(y)_i : mean and variance of y in bins of x
    This is effectively a ROOT.Profile
    """
    dig = numpy.digitize(x, bins)
    n = len(bins) - 1
    my, vy = numpy.zeros(n), numpy.zeros(n)

    for i in range(n):
        idx = (dig == i+1)

        if not idx.any(): #check for empty bin
            my[i] = numpy.nan
            vy[i] = numpy.nan
            continue

        if weights == None:
            my[i] = numpy.mean(y[idx])
            vy[i] = numpy.std(y[idx])**2
        else:
            my[i], vy[i] = meanAndVariance(y[idx], weights[idx])

    return my, vy
