import numpy


def binnedMean(x, y, xbins, weights=None):
    """
    <y>_i : mean of y in bins of x
    """
    dig = numpy.digitize(x, xbins)
    n = len(xbins) - 1
    my = numpy.zeros(n)
    for i in range(n):
        idx = (dig == i+1)
        if weights == None:
            my[i] = numpy.average(y[idx])
        else:
            my[i] = numpy.average(y[idx], weights=weights[idx])
    return my

def binnedMeanAndVariance(x, y, xbins, weights=None):
    """
    <y>_i, sigma(y)_i : mean and variance of y in bins of x
    """
    dig = numpy.digitize(x, xbins)
    n = len(xbins) - 1
    my, vy = numpy.zeros(n), numpy.zeros(n)
    for i in range(n):
        idx = (dig == i+1)
        if weights == None:
            my[i] = numpy.average(y[idx])
            vy[i] = numpy.std(y[idx])**2
        else:
            my[i] = numpy.average(y[idx], weights=weights[idx])
            vy[i] = numpy.dot((y[idx]-my[i])**2, weights[idx]) / sum(weights[idx])
    return (my, vy)
