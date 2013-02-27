
def nucleusID(A, Z):
    """ Given a mass and charge number, returns the nucleus ID. """
    return 1000000000 + Z * 10000 + A * 10

def nucleusID2Z(id):
    """ Given a nucleus ID, returns the charge number. """
    return id % 1000000000 // 10000

def nucleusID2A(id):
    """ Given a nucleus ID, returns the mass number. """
    return id % 10000 // 10
