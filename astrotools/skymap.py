"""
Module to visualize arrival directions or heatmaps on the sphere.
"""

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

from astrotools import coord


def scatter(v, c=None, cblabel='log$_{10}$(Energy / eV)', opath=None, **kwargs):
    """
    Scatter plot of events with arrival directions x,y,z and colorcoded energies.

    :param v: array of shape (3, n) pointing into directions of the events
    :param c: quantity that is supposed to occur in colorbar, e.g. energy of the cosmic rays
    :param cblabel: colorbar label
    :param opath: if not None, saves the figure to the given opath (no returns)
    :param kwargs: additional named keyword arguments

           - cmap: colormap
           - mask_alpha: alpha value for maskcolor
           - fontsize: scale the general fontsize
           - dark_grid: if True paints a dark grid (useful for bright maps)
           - gridcolor: Color of the grid.
           - gridalpha: Transparency value of the gridcolor.
           - tickcolor: Color of the ticks.
           - tickalpha: Transparency of the longitude ticks.
    :return: figure, axis of the scatter plot
    """

    lons, lats = coord.vec2ang(v)
    # mimic astronomy convention: positive longitudes evolving to the left with respect to GC
    lons = -lons

    fontsize = kwargs.pop('fontsize', 26)
    kwargs.setdefault('s', 8)
    kwargs.setdefault('lw', 0)
    if c is not None:
        finite = np.isfinite(c)
        vmin = kwargs.pop('vmin', smart_round(np.min(c[finite]), upper_border=False))
        vmax = kwargs.pop('vmax', smart_round(np.max(c[finite]), upper_border=True))

        step = smart_round((vmax - vmin) / 5., order=1)
        cticks = kwargs.pop('cticks', np.arange(vmin, vmax, step))

    # read keyword arguments for the grid
    dark_grid = kwargs.pop('dark_grid', True)
    gridcolor = kwargs.pop('gridcolor', 'lightgray' if dark_grid is None else 'black')
    gridalpha = kwargs.pop('gridalpha', 0.5 if dark_grid is None else 0.4)
    tickcolor = kwargs.pop('tickcolor', 'lightgray' if dark_grid is None else 'black')
    tickalpha = kwargs.pop('tickalpha', 0.5 if dark_grid is None else 1)

    # plot the events
    fig = plt.figure(figsize=[12, 6])
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.9], projection="hammer")
    events = ax.scatter(lons, lats, c=c, **kwargs)

    if c is not None:
        cbar = plt.colorbar(events, orientation='horizontal', shrink=0.85, pad=0.05, aspect=30,
                            cmap=kwargs.get('cmap'), ticks=cticks)
        cbar.set_label(cblabel, fontsize=fontsize)
        cbar.set_clim(vmin, vmax)
        cbar.ax.tick_params(labelsize=fontsize - 4)
        cbar.draw_all()

    # Setup the grid
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plot_grid(gridcolor=gridcolor, gridalpha=gridalpha, tickalpha=tickalpha, tickcolor=tickcolor, fontsize=fontsize)

    if opath is not None:
        plt.savefig(opath, bbox_inches='tight')
        plt.clf()

    return fig, ax


def heatmap(m, opath=None, label='entries', mask=None, maskcolor='white', **kwargs):
    """
    Heatmap plot of binned data m. For exmaple usage see: cosmic_rays.plot_healpy_map()

    :param m: Array with size npix for an arbitrary healpy nside.
    :param opath: if not None, saves the figure to the given opath (no returns)
    :param label: label for the colormap
    :param mask: either boolean mask that paints certain pixels different or condition for m
    :param maskcolor: which color to paint the mask
    :param kwargs: additional named keyword arguments

           - cmap: colormap
           - mask_alpha: alpha value for maskcolor
           - fontsize: scale the general fontsize
           - xresolution: Scales the resolution of the plot (default: 800)
           - width: Size of the figure
           - dark_grid: if True paints a dark grid (useful for bright maps)
           - gridcolor: Color of the grid.
           - gridalpha: Transparency value of the gridcolor.
           - tickcolor: Color of the ticks.
           - tickalpha: Transparency of the longitude ticks.
    :return: figure of the heatmap, colorbar
    """

    # read general keyword arguments
    fontsize = kwargs.pop('fontsize', 26)

    # create the meshgrid and project the map to a rectangular matrix (xresolution x yresolution)
    xresolution = kwargs.pop('xresolution', 800)
    yresolution = xresolution // 2
    theta = np.linspace(np.pi, 0, yresolution)
    phi = np.linspace(-np.pi, np.pi, xresolution)
    longitude = np.radians(np.linspace(-180, 180, xresolution))
    latitude = np.radians(np.linspace(-90, 90, yresolution))

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(hp.get_nside(m), theta_grid, phi_grid)

    # management of the colormap
    cmap = kwargs.pop('cmap', 'viridis')
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    if mask is not None:
        if not hasattr(mask, 'size'):
            mask = m == mask
        m = np.ma.masked_where(mask, m)
        cmap.set_bad(maskcolor, alpha=kwargs.pop('mask_alpha', 1))
    grid_map = m[grid_pix]

    finite = np.isfinite(m)
    vmin = kwargs.pop('vmin', smart_round(np.min(m[finite]), upper_border=False))
    vmax = kwargs.pop('vmax', smart_round(np.max(m[finite]), upper_border=True))

    # read keyword arguments for the grid
    dark_grid = kwargs.pop('dark_grid', None)
    gridcolor = kwargs.pop('gridcolor', 'lightgray' if dark_grid is None else 'black')
    gridalpha = kwargs.pop('gridalpha', 0.5 if dark_grid is None else 0.4)
    tickcolor = kwargs.pop('tickcolor', 'lightgray' if dark_grid is None else 'black')
    tickalpha = kwargs.pop('tickalpha', 0.5 if dark_grid is None else 1)

    # Start plotting
    width = kwargs.pop('width', 12)
    fig = plt.figure(figsize=(width, width))
    fig.add_subplot(111, projection='hammer')
    # flip longitude to the astro convention
    # rasterized makes the map bitmap while the labels remain vectorial
    image = plt.pcolormesh(longitude[::-1], latitude, grid_map, rasterized=True, vmin=vmin,
                           vmax=vmax, cmap=cmap, edgecolor='face', **kwargs)
    cb = fig.colorbar(image, ticks=[vmin, (vmin + vmax) / 2, vmax], format='%g',
                      orientation='horizontal', aspect=30, shrink=0.9, pad=0.05)
    cb.solids.set_edgecolor("face")
    cb.set_label(label, fontsize=30)
    cb.ax.tick_params(axis='x', direction='in', size=3, labelsize=26)

    # Setup the grid
    plot_grid(gridcolor=gridcolor, gridalpha=gridalpha, tickalpha=tickalpha, tickcolor=tickcolor, fontsize=fontsize)

    if opath is not None:
        plt.savefig(opath, bbox_inches='tight')
        plt.clf()

    return fig, cb


def plot_grid(lon_ticks=None, lat_ticks=None, lon_grid=30, lat_grid=30, fontsize=26, **kwargs):
    """
    Plot a grid on the skymap.

    :param lon_ticks: Set the label ticks for the longitudes (default: [90, 0, -90]).
    :param lat_ticks: Set the label ticks for the latitude (default: [-60, -30, 0, 30, 60]).
    :param lon_grid: Distances between the grid lines in longitude in degrees (default: 30 deg).
    :param lat_grid: Distances between the grid lines in latitude in degrees (default: 30 deg).
    :param kwargs: additional named keyword arguments

            - gridcolor: Color of the grid.
            - gridalpha: Transparency value of the gridcolor.
            - tickcolor: Color of the ticks.
            - tickalpha: Transparency of the longitude ticks.
    """
    lon_ticks = [90, 0, -90] if lon_ticks is None else lon_ticks
    lat_ticks = [-60, -30, 0, 30, 60] if lat_ticks is None else lat_ticks
    plt.gca().set_longitude_grid(lon_grid)
    plt.gca().set_latitude_grid(lat_grid)
    plt.gca().set_longitude_grid_ends(89)

    plt.grid(alpha=kwargs.pop('gridalpha', 0.5), color=kwargs.pop('gridcolor', 'lightgray'))
    plt.gca().set_xticklabels(np.hstack([(r'', r'', r'%d$^{\circ}$' % lon) for lon in lon_ticks]),
                              alpha=kwargs.pop('tickalpha', 0.5), fontsize=fontsize)
    plt.gca().tick_params(axis='x', colors=kwargs.pop('tickcolor', 'lightgray'))
    plt.gca().set_yticklabels([r'%d$^{\circ}$' % lat for lat in lat_ticks], fontsize=fontsize)


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


def skymap(m, **kwargs):
    """ Deprecated funcion -> See heatmap() """
    print("User warning: function skymap() is deprecated and will stop existing in v2.0.0. \
            Please use heatmap() in future!")
    return heatmap(m, **kwargs)


def healpy_map(m, **kwargs):
    """ Forwards to function heatmap() """
    return heatmap(m, **kwargs)


def eventmap(v, **kwargs):
    """ Forwards to function scatter() """
    return scatter(v, **kwargs)
