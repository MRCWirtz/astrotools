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

def pix2Ang(i, nest=False):
    """
    Convert HEALpixel i to spherical angles (astrotools definition)
    Substitutes healpy.pix2ang
    """
    print 'Not implemented'
    return (0, 0)

def ang2Pix(phi, theta, nest=False):
    """
    Convert spherical angle (astrotools definition) to HEALpixel i
    Substitutes healpy.ang2pix
    """
    print 'Not implemented'
    return 0


if __name__ == "__main__":
    import matplotlib.pyplot
    import coord

    # pixel 5 in base resolution (n = 0)
    n = 0
    iPix = 5
    v = healpy.pix2vec(2**n, iPix, nest=True)
    phi, theta = coord.vec2Ang(*v)

    # centers of four-fold upsampled pixels
    nside_up = 2**(n+4)
    iPix_up = range(iPix * 4**4, (iPix+1) * 4**4)
    x, y, z = healpy.pix2vec(nside_up, iPix_up, nest=True)
    phi_up, theta_up = coord.vec2Ang(x, y, z)

    # 20 random direction within the pixel
    v = randVecInPix(n, numpy.ones(20, dtype=int) * iPix)
    phi_rnd, theta_rnd = coord.vec2Ang(*v)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.plot(phi_up, theta_up, 'b+')
    ax.plot(phi_rnd, theta_rnd, 'go')
    ax.plot(phi, theta, 'ro')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
