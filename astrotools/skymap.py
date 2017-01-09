"""
Skyplots
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy
import astrotools.coord as coord


def scatter(v, E=None):
    """
    Scatter plot of events with arrival directions x,y,z and colorcoded energies.
    """
    if E == None:
        E = np.ones(np.shape(v)[1])
    energies = np.log10(E) + 18
    lons, lats = coord.vec2ang(v)

    # sort by energy
    idx = np.argsort(energies)
    energies = energies[idx]
    lons = lons[idx]
    lats = lats[idx]

    lons = -lons  # mimick astronomy convention

    fig = plt.figure(figsize=[12,6])
    ax = fig.add_axes([0.1,0.1,0.85,0.9], projection = "hammer")
    events = ax.scatter(lons, lats, c=energies, lw=0, s=8, vmin=18.5, vmax=20.5)

    cbar = plt.colorbar(events, orientation='horizontal', pad=0.05, aspect=40)
    cbar.set_label("$\log_{10}$(Energy/[eV])")
    cbar.set_ticks(np.arange(18.5, 20.6, 0.5))

    ax.set_xticks(np.deg2rad(np.linspace(-120,120,5)))
    ax.set_xticklabels(())
    ax.set_yticks(np.deg2rad(np.linspace(-60,60,5)))
    ax.grid(True)
    return fig



def skymap(m, xsize=500, width=12, **kwargs):
    nside = healpy.npix2nside(len(m))

    ysize = xsize / 2

    theta = np.linspace(np.pi, 0, ysize)
    phi   = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))

    # project the map to a rectangular matrix xsize x ysize
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix = healpy.ang2pix(nside, THETA, PHI)
    grid_map = m[grid_pix]


    fig = plt.figure(figsize=(width, width))
    ax = fig.add_subplot(111, projection='mollweide')

    # rasterized makes the map bitmap while the labels remain vectorial
    # flip longitude to the astro convention
    image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=vmin, vmax=vmax, rasterized=True)#, cmap=cmap)

    # graticule
    ax.set_longitude_grid(60)
    ax.set_latitude_grid(30)
    # ax.tick_params(axis='x', labelsize=10)
    # ax.tick_params(axis='y', labelsize=10)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    plt.grid(True)
