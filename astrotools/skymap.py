import numpy
import matplotlib.pyplot as plt
from astrotools import coord, healpytools
import healpy


def scatter(x, y, z, E):
    """
    Scatter plot of events with arrival directions x,y,z colorcoded energies.
    """
    energies = numpy.log10(E) + 18
    lons, lats = coord.vec2Ang(x, y, z)

    # sort by energy
    idx = numpy.argsort(energies)
    energies = energies[idx]
    lons = lons[idx]
    lats = lats[idx]

    fig = plt.figure(figsize=[12,6])
    ax = fig.add_axes([0.1,0.1,0.85,0.9], projection = "hammer")
    events = ax.scatter(lons, lats, c=energies, lw=0, s=8, vmin=18.5, vmax=20.5)

    cbar = plt.colorbar(events, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label("$\log_{10}$(Energy/[eV])")
    cbar.set_ticks(numpy.arange(18.5, 20.6, 0.5))

    ax.set_xticks(numpy.deg2rad(numpy.linspace(-120,120,5)))
    ax.set_xticklabels(())
    ax.set_yticks(numpy.deg2rad(numpy.linspace(-60,60,5)))
    ax.grid(True)
    return fig

def statistic(nside, x, y, z, statistic='count', vals=None, **kwargs):
    """
    Mollweide projection of a given statistic per HEALpixel.

    Parameters
    ----------
    x, y, z: array_like
        coordinates
    statistic: keyword
        'number', 'frequency', 'mean' or 'rms'
    vals: array_like
        values for which the mean or rms is calculated
    """
    npix = healpy.nside2npix(nside)
    pix = healpy.vec2pix(nside, x, y, z)
    bins = numpy.arange(npix + 1) - 0.5
    nmap = numpy.histogram(pix, bins=bins)[0] # count per pixel

    if statistic == 'count':
        vmap = nmap.astype('float')
    elif statistic == 'frequency':
        vmap = nmap.astype('float') / max(nmap) # frequency normalized to maximum
    elif statistic == 'mean':
        if vals == None: raise ValueError
        vmap = numpy.histogram(pix, weights=vals, bins=bins)[0]
        vmap /= nmap # mean
    elif statistic == 'rms':
        if vals == None: raise ValueError
        vmap = numpy.histogram(pix, weights=vals**2, bins=bins)[0]
        vmap = (vmap / nmap)**.5 # rms

    vmap[nmap==0] = healpy.UNSEEN
    healpy.mollview(vmap, **kwargs)

