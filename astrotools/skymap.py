"""
Skyplots
"""
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import astrotools.coord as coord


def scatter(v, log10e, cblabel='log$_{10}$(Energy / eV)', fontsize=26, opath=None, **kwargs):
    """
    Scatter plot of events with arrival directions x,y,z and colorcoded energies.

    :param v: array of shape (3, n) pointing into directions of the events
    :param log10e: energy of the cosmic rays in log10(E / eV)
    :return: figure of the scatter plot
    """

    lons, lats = coord.vec2ang(v)

    # sort by energy
    idx = np.argsort(log10e)
    log10e = log10e[idx]
    lons = lons[idx]
    lats = lats[idx]

    lons = -lons  # mimic astronomy convention

    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.9], projection="hammer")
    kwargs.setdefault('s', 8)
    kwargs.setdefault('lw', 0)
    kwargs.setdefault('vmin', np.min(log10e))
    kwargs.setdefault('vmax', np.max(log10e))
    events = ax.scatter(lons, lats, c=log10e, **kwargs)

    cbar = plt.colorbar(events, orientation='horizontal', shrink=0.85, pad=0.05, aspect=30, cmap=kwargs.get('cmap'))
    cbar.set_label(cblabel, fontsize=fontsize)
    step = 0.2 if np.max(log10e) - np.min(log10e) > 1. else 0.1
    cbar.set_ticks(np.arange(round(np.min(log10e), 1), round(np.max(log10e), 1), step))
    cbar.ax.tick_params(labelsize=fontsize - 4)

    plt.xticks(np.arange(-5. / 6. * np.pi, np.pi, np.pi / 6.),
               ['', '', r'90$^{\circ}$', '', '', r'0$^{\circ}$', '', '', r'-90$^{\circ}$', '', ''], fontsize=fontsize)
    # noinspection PyTypeChecker
    plt.yticks([-np.radians(60), -np.radians(30), 0, np.radians(30), np.radians(60)],
               [r'-60$^{\circ}$', r'-30$^{\circ}$', r'0$^{\circ}$', r'30$^{\circ}$', r'60$^{\circ}$'],
               fontsize=fontsize)

    ax.grid(True)

    if opath is not None:
        plt.savefig(opath, bbox_inches='tight')
        plt.clf()


def smart_round(v, order=2, upper_border=True):
    """
    Rounds a value v such that it can be used e.g. for colorbars

    :param v: scalar value which should be rounded
    :type v: int, float
    :param upper_border: round such that the value can be used as an upper border of an interval, default=True
    :param order: number of digits to round to, default=2
    :return: rounded value
    :rtype: int, float

    This function has been tested on the following numbers (with all upper_border presented here):

    .. code-block:: python

        :linenos:
        >> from plotting import smart_round
        >> smart_round(100000), smart_round(100000, upper_border=False)
        100000.0, 100000.0
        >> smart_round(100001), smart_round(100001, upper_border=False)
        101000.0, 100000.0
        >> smart_round(-100001), smart_round(-100001, upper_border=False)
        -100000.0, -100000.0
        >> smart_round(2.23), smart_round(2.23, upper_border=False)
        2.23, 2.23
        >> smart_round(2.230), smart_round(2.230, upper_border=False)
        2.23, 2.23
        >> smart_round(2.231), smart_round(2.231, upper_border=False)
        2.24, 2.23
        >> smart_round(-2.230), smart_round(-2.230, upper_border=False)
        -2.23, -2.23
        >> smart_round(-2.231), smart_round(-2.231, upper_border=False)
        -2.23, -2.24
        >> smart_round(0.930001), smart_round(0.930001, upper_border=False)
        0.94, 0.93
        >> smart_round(-0.930001), smart_round(-0.930001, upper_border=False)
        -0.93, -0.94
    """
    if v == 0:
        return 0
    o = np.log10(np.fabs(v))
    f = 10 ** (-int(o) + order)
    if upper_border:
        return np.ceil(v * f) / f
    return np.floor(v * f) / f


def plot_grid(xangles=None, yangles=None, gridcolor='lightgray', gridalpha=0.5,
              grid_tick_alpha=0.5, grid_tick_color='lightgray'):
    """Plot a grid on the skymap"""
    if xangles is None:
        xangles = [270, 180, 90]
    if yangles is None:
        yangles = [-60, -30, 0, 30, 60]
    plt.gca().set_longitude_grid(30)
    plt.gca().set_latitude_grid(30)
    plt.gca().set_longitude_grid_ends(89)

    plt.grid(alpha=gridalpha, color=gridcolor)
    plt.gca().set_xticklabels([r'',
                               r'',
                               r'%d$^{\circ}$' % xangles[0],
                               r'',
                               r'',
                               r'%d$^{\circ}$' % xangles[1],
                               r'',
                               r'',
                               r'%d$^{\circ}$' % xangles[2],
                               r'',
                               r''], alpha=grid_tick_alpha)
    plt.gca().tick_params(axis='x', colors=grid_tick_color)
    plt.gca().set_yticklabels([r'%d$^{\circ}$' % yangles[0],
                               r'%d$^{\circ}$' % yangles[1],
                               r'%d$^{\circ}$' % yangles[2],
                               r'%d$^{\circ}$' % yangles[3],
                               r'%d$^{\circ}$' % yangles[4]])


def skymap(m, opath=None, label='entries', fontsize=26, xsize=500, width=12, mask=None, mask_alpha=1, **kwargs):
    """Plot a skymap"""
    cmap = kwargs.pop("cmap", 'viridis')
    maskcolor = kwargs.pop("maskcolor", 'white')
    dark_grid = kwargs.pop("dark_grid", None)
    nside = hp.get_nside(m)
    ysize = xsize // 2

    theta = np.linspace(np.pi, 0, ysize)
    phi = np.linspace(-np.pi, np.pi, xsize)
    longitude = np.radians(np.linspace(-180, 180, xsize))
    latitude = np.radians(np.linspace(-90, 90, ysize))

    # project the map to a rectangular matrix xsize x ysize
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(nside, theta_grid, phi_grid)
    grid_map = m[grid_pix]

    fig = plt.figure(figsize=(width, width))
    fig.add_subplot(111, projection='hammer')

    # rasterized makes the map bitmap while the labels remain vectorial
    # flip longitude to the astro convention
    finite = np.isfinite(m)
    vmin = kwargs.get('vmin', smart_round(np.min(m[finite])))
    vmax = kwargs.get('vmin', smart_round(np.max(m[finite])))

    # deal with colormaps
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)

    if mask is not None:
        if not hasattr(mask, 'size'):
            mask = m == mask
        m = np.ma.masked_where(mask, m)
        cmap.set_bad(maskcolor, alpha=mask_alpha)

    image = plt.pcolormesh(longitude[::-1], latitude, grid_map, vmin=vmin, vmax=vmax, rasterized=True,
                           antialiased=False, edgecolor='face', cmap=cmap, **kwargs)
    cb = fig.colorbar(image, ticks=[vmin, (vmin + vmax) / 2, vmax], format='%g',
                      orientation='horizontal', aspect=30, shrink=0.9, pad=0.05)
    cb.solids.set_edgecolor("face")
    cb.set_label(label, fontsize=30)
    cb.ax.tick_params(axis='x', direction='in', size=3, labelsize=26)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # Setup the grid
    if dark_grid is None:
        plot_grid(xangles=[90, 0, -90])
    else:
        plot_grid(xangles=[90, 0, -90], gridcolor='black', gridalpha=0.4, grid_tick_alpha=1, grid_tick_color='black')

    if opath is not None:
        plt.savefig(opath, bbox_inches='tight')
        plt.clf()
