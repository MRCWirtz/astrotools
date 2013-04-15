from numpy import *


def binnedMean(x, y, xbins, weights=None):
    """
    <y>_i : mean of y in bins of x
    """
    dig = digitize(x, xbins)
    n = len(xbins) - 1
    my = zeros(n)
    for i in range(n):
        idx = (dig == i+1)
        if weights == None:
            my[i] = average(y[idx])
        else:
            my[i] = average(y[idx], weights=weights[idx])
    return my

def binnedMeanAndVariance(x, y, xbins, weights=None):
    """
    <y>_i, sigma(y)_i : mean and standard deviation (spread) of y in bins of x
    """
    dig = digitize(x, xbins)
    n = len(xbins) - 1
    my, vy = zeros(n), zeros(n)
    for i in range(n):
        idx = (dig == i+1)
        if weights == None:
            my[i] = mean(y[idx])
            vy[i] = std(y[idx])**.5
        else:
            my[i] = average(y[idx], weights=weights[idx])
            vy[i] = dot((y[idx]-my[i])**2, weights[idx]) / sum(weights[idx])
    return (my, vy)
