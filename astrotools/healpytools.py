from numpy import *
import healpy as h


def upsample(n_up, i):
    '''
    Upsample a healpix pixel from order n to (n + n_up).

    Parameters
    ----------
    n_up : int
    order of upsampling (upsampling to 4^n_up subpixels)
    i : int
    pixel number

    Returns
    -------
    i_up : int, array
    pixel numbers of subpixels in nested healpix scheme
    '''
    i_up = range(i*4**n_up, (i+1)*4**n_up)
    return i_up


def randUniformInPixel(n, i):
    '''
    Draw random directions within a pixel.

    Parameters
    ----------
    n : int
    healpix order (nside = 2^n, npix = 12 * 4^n)
    i : int or array-like
    pixel number(s)

    Returns
    -------
    theta, phi : float, scalar or array-like
    The angular coordinates of the random directions drawn within the pixel(s) i
    '''
    s = size(i)
    n_up = 29 - n # order of upsampling, 29 is the the maximum healpix order using 64 bit ints
    i_up = i * 4**n_up + random.randint(0, 4**n_up, size=s)
    phi, theta = h.pix2ang(2**29, i_up, nest=True)
    return phi, theta


if __name__ == "__main__":
    import matplotlib.pyplot as mpl
    # pixel 5 in base resolution (n = 0)
    n = 0
    iPix = 5
    phi, theta = h.pix2ang(2**n, iPix, nest=True)

    # centers of for-fold upsampled pixel
    phi_up, theta_up = h.pix2ang(2**(n+4), upsample(4, iPix), nest=True)

    # 20 random direction within the pixel
    phi_rd, theta_rd = randUniformInPixel(n, ones(20, dtype=int) * iPix)

    fig = mpl.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.plot(phi_up, pi/2-theta_up, 'b+')
    ax.plot(phi_rd, pi/2-theta_rd, 'go')
    ax.plot(phi, pi/2-theta, 'ro')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()
