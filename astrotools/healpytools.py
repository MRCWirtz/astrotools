# Extensions to healpy

import numpy
import healpy


def randVecInPix(nside, ipix, nest=False):
    """
    Draw vectors from a uniform distribution within a HEALpixel.
    nside : healpix nside parameter
    ipix  : pixel number(s)
    """
    if not(nest):
        ipix = healpy.ring2nest(nside, ipix=ipix)

    norder = nside2norder(nside)
    print norder
    nUp = 29 - norder
    iUp = ipix * 4**nUp

    if numpy.iterable(ipix):
        iUp += numpy.random.randint(0, 4**nUp, size=numpy.size(ipix))
    else:
        iUp += numpy.random.randint(0, 4**nUp)
    vec = healpy.pix2vec(nside=2**29, ipix=iUp, nest=True)
    return vec

def pix2ang(i, nest=False):
    """
    Convert HEALpixel i to spherical angles (astrotools definition)
    Substitutes healpy.pix2ang
    """
    print 'Not implemented'
    return (0, 0)

def ang2pix(phi, theta, nest=False):
    """
    Convert spherical angle (astrotools definition) to HEALpixel i
    Substitutes healpy.ang2pix
    """
    print 'Not implemented'
    return 0

def angularDistance(nside, i, j, nest=False):
    """
    Give the angular distance between two pixel.
    """
    v1 = healpy.pix2vec(nside, i, nest)
    v2 = healpy.pix2vec(nside, j, nest)
    return numpy.arccos( numpy.dot(v1, v2) )

def norder2npix(norder):
    """
    Give the number of pixel for the given HEALpix order.
    """
    return 12*4**norder

def npix2norder(npix):
    """
    Give the HEALpix order for the given number of pixel.
    """
    norder = numpy.log(npix/12) / numpy.log(4)
    if not(norder.is_integer()):
        raise ValueError('Wrong pixel number (it is not 12*4**norder)')
    return int(norder)

def norder2nside(norder):
    """
    Give the HEALpix nside parameter for the given HEALpix order.
    """
    return 2**norder

def nside2norder(nside):
    """
    Give the HEALpix order for the given HEALpix nside parameter.
    """
    norder = numpy.log2(nside)
    if not(norder.is_integer()):
        raise ValueError('Wrong nside number (it is not 2**norder)')
    return int(norder)


if __name__ == "__main__":
    import matplotlib.pyplot
    import coord

    # pixel 5 in base resolution (n = 0)
    norder = 0
    iPix = 5
    v = healpy.pix2vec(norder2nside(norder), iPix, nest=True)
    phi, theta = coord.vec2Ang(*v)

    # centers of four-fold upsampled pixels
    nside_up = 2**(norder + 4)
    iPix_up = range(iPix * 4**4, (iPix+1) * 4**4)
    x, y, z = healpy.pix2vec(nside_up, iPix_up, nest=True)
    phi_up, theta_up = coord.vec2Ang(x, y, z)

    # 20 random direction within the pixel
    v = randVecInPix(norder2nside(norder), numpy.ones(20, dtype=int) * iPix)
    phi_rnd, theta_rnd = coord.vec2Ang(*v)

    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.plot(phi_up, theta_up, 'b+')
    ax.plot(phi_rnd, theta_rnd, 'go')
    ax.plot(phi, theta, 'ro')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
