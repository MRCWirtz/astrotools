# Tools for CRPropa 3
import numpy as np


def nucleusID(A, Z):
    """ Given a mass and charge number, returns the nucleus ID (2006 PDG standard). """
    return 1000000000 + Z * 10000 + A * 10

def nucleusID2Z(pid):
    """ Given a nucleus ID (2006 PDG standard), returns the charge number. """
    return pid % 1000000000 // 10000

def nucleusID2A(pid):
    """ Given a nucleus ID (2006 PDG standard), returns the mass number. """
    return pid % 10000 // 10


def iter_loadtxt(filename, delimiter='\t', skiprows=0, dtype=float, unpack=False):
    """
    Lightweight loading function for large tabulated data files in ASCII format.
    Memory requirement is greatily reduced compared to np.genfromtxt and np.loadtxt.
    """
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    if unpack:
        return data.transpose()
    return data

#from pxl import core, astro
#import numpy

#def Events1d(A, container):
#  # PDG_Code	Energy[EeV]	Age[Mpc]Initial_PDG_Code	Initial_Energy[EeV]
#  ID, E, AGE, ID0, E0 = A.transpose()
#  for i in range(len(ID)):
#    cr = astro.UHECR()
#    cr.setEnergy(E[i])
#    cr.setCharge(int((ID[i]-1e9)//10000))
#    cr.setMass(int((ID[i]%1000)//10))
#    cr.setUserRecord('E0', float(E0[i]))
#    cr.setUserRecord('Z0', int((ID0[i]-1e9)//10000))
#    cr.setUserRecord('A0', int((ID0[i]%1000)//10))
#    container.setObject(cr)
#  return container


#def Events3d(A, container):
#  # PDG_Code Energy[EeV] Position(X,Y,Z)[Mpc] Direction(Phi,Theta) Age[Mpc] Initial_PDG_Code Initial_Energy[EeV] Initial_Position(X,Y,Z)[Mpc] Initial_Direction(Phi,Theta)
#  ID, E, X, Y, Z, PHI, THETA, AGE, ID0, E0, X0, Y0, Z0, PHI0, THETA0 = A.transpose()
#  p, x1, x2 = core.Basic3Vector(), core.Basic3Vector(), core.Basic3Vector()

#  for i in range(len(ID)):
#    cr = astro.UHECR()

#    # set arrival direction, correcting for the geometric distortion from an extended observer
#    p.setRThetaPhi( 1, THETA[i], PHI[i] )
#    x1.setXYZ( X[i]-X0[i], Y[i]-Y0[i], Z[i]-Z0[i] )
#    x2.setXYZ( 118.34-X0[i], 117.69-Y0[i], 119.2-Z0[i] )
#    p.rotate( x1.cross(x2), x1.getAngleTo(x2) )
#    # we look from inside the celestial sphere
#    cr.setSuperGalacticDirectionVector( -p )

#    cr.setEnergy(E[i])
#    cr.setCharge(int((ID[i]-1e9)//10000))
#    cr.setMass(int((ID[i]%1000)//10))

#    cr.setUserRecord('E0', float(E0[i]))
#    cr.setUserRecord('Z0', int((ID0[i]-1e9)//10000))
#    cr.setUserRecord('A0', int((ID0[i]%1000)//10))
#    cr.setUserRecord('x0', x2.getX())
#    cr.setUserRecord('y0', x2.getY())
#    cr.setUserRecord('z0', x2.getZ())
#    container.setObject(cr)
#  return container


#def saveToPxlio(ifname, ofname=None):
#  '''
#  Converts a mpc .txt to a .pxlio file.
#  The events in the input file are converted to pxl.astro.UHECRs and stored in a pxl.core.BasicContainer.
#  The events are shuffled.
#  Input coordinates are assumed to be supergalactic.
#  
#  Parameters
#  ----------
#  ifname : string
#           path to a mpc txt input file
#  ofname : string, optional
#           name of the pxlio output file
#  '''
#  print 'Reading file', ifname
#  A = numpy.genfromtxt(ifname)
#  numpy.random.shuffle(A) # shuffle events

#  container = core.BasicContainer()

#  nLines, nColumns = numpy.shape(A)
#  print nLines, 'lines, ', nColumns, 'columns'

#  if nColumns == 15:
#    print '3d events'
#    Events3d(A, container)
#  elif nColumns == 5:
#    print '1d events'
#    Events1d(A, container)
#  else:
#    print 'Invalid number of columns'
#    return False

#  if ofname == None:
#    ofname = ifname.replace('.txt', '.pxlio')
#  fout = core.OutputFile(ofname)
#  fout.writeBasicContainer(container)
#  fout.close()
