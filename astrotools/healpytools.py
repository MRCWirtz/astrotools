import numpy
import healpy


def randVecInPix(nOrder, iPix, nest=False):
    """
    Draw vectors from a uniform distribution within a HEALpixel.

    n :    healpix order (nside = 2^n, npix = 12 * 4^n)
    iPix : pixel number(s)
    """
    if not(nest):
        iPix = healpy.ring2nest(nside=2**nOrder, ipix=iPix)
    nUp = 29 - nOrder
    iUp = iPix * 4**nUp
    if numpy.iterable(iPix):
        iUp += numpy.random.randint(0, 4**nUp, size=numpy.size(iPix))
    else:
        iUp += numpy.random.randint(0, 4**nUp)
    vec = healpy.pix2vec(nside=2**29, ipix=iUp, nest=True)
    return vec


if __name__ == "__main__":
    import matplotlib.pyplot
    import coordinates

    # pixel 5 in base resolution (n = 0)
    n = 0
    iPix = 5
    v = healpy.pix2vec(2**n, iPix, nest=True)
    r, phi, theta = coordinates.cartesian2Spherical(*v)

    # centers of four-fold upsampled pixels
    nside_up = 2**(n+4)
    iPix_up = range(iPix * 4**4, (iPix+1) * 4**4)
    x, y, z = healpy.pix2vec(nside_up, iPix_up, nest=True)
    r, phi_up, theta_up = coordinates.cartesian2Spherical(x, y, z)

    # 20 random direction within the pixel
    v = randVecInPix(n, numpy.ones(20, dtype=int) * iPix)
    r, phi_rnd, theta_rnd = coordinates.cartesian2Spherical(*v)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.plot(phi_up, theta_up, 'b+')
    ax.plot(phi_rnd, theta_rnd, 'go')
    ax.plot(phi, theta, 'ro')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
