# Tools for CRPropa 3
# Particle IDs conform to the 2006 PDG standard.

def nucleusID(A, Z):
    """ Given a mass and charge number, returns the nucleus ID. """
    return 1000000000 + Z * 10000 + A * 10

def nucleusID2Z(pid):
    """ Given a nucleus ID, returns the charge number. """
    return pid % 1000000000 // 10000

def nucleusID2A(pid):
    """ Given a nucleus ID, returns the mass number. """
    return pid % 10000 // 10
