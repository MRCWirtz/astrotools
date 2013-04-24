
def nucleusID(A, Z):
    """ Given a mass and charge number, returns the nucleus ID. """
    return 1000000000 + Z * 10000 + A * 10

def nucleusID2Z(id):
    """ Given a nucleus ID, returns the charge number. """
    return id % 1000000000 // 10000

def nucleusID2A(id):
    """ Given a nucleus ID, returns the mass number. """
    return id % 10000 // 10

#def removeFamilies(data):
#    """
#    Removes 
#    """
#    families = defaultdict(list)
#    for i in range(len(data)):
#        key = data[i]['E0'], data[i]['P0x'], data[i]['P0y'], data[i]['P0z'], data[i]['X0'], data[i]['Y0'], data[i]['Z0']
#        families[key].append(i)
#    for key in families.keys():
#        f = families[keys]
#        f.pop(randint(0, len(f)-1))

#    toRemove = list(flatten(families.values()))
#    print ("Removing", len(toRemove), "of", len(data))
#    return delete(data, toRemove)
