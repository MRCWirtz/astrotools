# -*- coding: utf-8 -*-
import numpy as np

__author__ = 'Martin Urban'

_dtype_template = [("C", float), ("pixel", int), ("lon", float), ("lat", float), ("log10e", float), ("xmax", float)]


def join_struct_arrays(arrays):
    """
    A function to join a list of numpy named arrays. An alternative (which can be slower) could be:
    # import numpy.lib.recfunctions as rfn
    # rfn.merge_arrays((d,e), flatten = True, usemask = False)
    numpy.lib.recfunctions as rfn is a collection of utilities to manipulate structured arrays.
    Documentation on recfunctions with examples can be found here:
    http://pyopengl.sourceforge.net/pydoc/numpy.lib.recfunctions.html
    The following code is taken from:
    http://stackoverflow.com/questions/5355744/numpy-joining-structured-arrays

    :param arrays: a tuple or list of arrays which should be joined
    :return: an array containing the joined arrays
    :rtype: numpy named array
    """
    sizes = np.array([a.itemsize for a in arrays])
    offsets = np.r_[0, sizes.cumsum()]
    n = len(arrays[0])
    joint = np.empty((n, offsets[-1]), dtype=np.uint8)
    for a, size, offset in zip(arrays, sizes, offsets):
        joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)
    dtype = sum((a.dtype.descr for a in arrays), [])
    return joint.ravel().view(dtype)


def change_nametype2object(data, name_to_be_retyped, new_type=object):
    """
    changes the type of a part of the array,
    for examples see https://gist.github.com/d95c9f604f2fc8594ccbe47534302b24.git

    :param data: numpy recarray
    :param name_to_be_retyped: name of the part which should be changed
    :param new_type: new type, must be something which can be converted into a numpy.dtype
    :return: data with changed type
    """
    new_dtype = []
    for name, i in zip(data.dtype.names, range(len(data.dtype))):
        if name == name_to_be_retyped:
            dt = new_type
        else:
            dt = data.dtype[i]
        new_dtype.append((name, dt))

    return data.astype(np.dtype(new_dtype))


class CosmicRaysBase:
    def __init__(self, cosmic_rays=None):
        self.type = "CosmicRays"
        self.general_object_store = {}
        if cosmic_rays is None:
            raise NotImplementedError(
                "Either the number of cosmic rays has to be set or the numpy array with cosmic rays has to be given"
                "or a filename to load cosmic rays from has to be given")
        if isinstance(cosmic_rays, str):
            self.load(cosmic_rays)
        elif isinstance(cosmic_rays, (int, np.dtype)):
            dtype_template = _dtype_template if isinstance(cosmic_rays, int) else cosmic_rays
            ncrs = cosmic_rays if isinstance(cosmic_rays, int) else 0
            cosmic_ray_template = np.zeros(shape=ncrs, dtype=dtype_template)
            self.cosmic_rays = cosmic_ray_template
        else:
            try:
                if cosmic_rays.type == "CosmicRays":
                    self.__copy__(cosmic_rays)
            except AttributeError:
                try:
                    self.cosmic_rays = cosmic_rays
                except NotImplementedError:
                    raise NotImplementedError("Trying to instantiate the CosmicRays class with a non "
                                              "supported type of cosmic_rays")
        self.ncrs = len(self.cosmic_rays)  # type: int
        self.keys = self.get_keys()
        # self.additional_elements = set()  # not used anymore because of we add everything to the array if possible
        self._create_access_functions()

    def __getitem__(self, key):
        if key in self.general_object_store.keys():
            return self.general_object_store[key]
        else:
            try:
                return self.cosmic_rays[key]
            except ValueError as e:
                raise ValueError("The key %s does not exist and the error message was %s" % (key, str(e)))

    def __setitem__(self, key, value):
        if key in self.cosmic_rays.dtype.names:
            self.cosmic_rays[key] = value
            return
        try:
            all_crs = len(value)
        except TypeError:
            all_crs = False
        if all_crs == self.ncrs:
            if isinstance(value[0], (list, float, str)):
                self.cosmic_rays = join_struct_arrays(
                    [self.cosmic_rays, np.array(value, dtype=[(key, type(value[0]))])])
            else:
                tmp = np.zeros(self.ncrs, dtype=[(key, float)])
                self.cosmic_rays = join_struct_arrays([self.cosmic_rays, tmp])
                self.cosmic_rays = change_nametype2object(self.cosmic_rays, key, object)
                self.cosmic_rays[key] = value
            self.__dict__.update({key: self._fun_factory(key)})
        else:
            try:
                self.general_object_store[key] = value
                self.__dict__.update({key: self._fun_factory(key)})
            except KeyError as e:
                raise KeyError("This key can not be set and the error message was %s" % str(e))
            except ValueError as e:
                raise KeyError("This value can not be set and the error message was %s" % str(e))
            except Exception as e:
                raise NotImplementedError("An unforeseen error happened: %s" % str(e))

    def __len__(self):
        return self.ncrs

    def __copy__(self, crs):
        """
        Function allows to copy a cosmic ray object to another object

        :param crs: instance of CosmicRays class 
        """
        self.cosmic_rays = crs.get_array().copy()
        self._update_attributes()
        for key in crs.get_keys():
            if key not in self.cosmic_rays.dtype.names:
                self.__setitem__(key, crs[key])

    def _update_attributes(self):
        self.ncrs = len(self.cosmic_rays)

    def _create_access_functions(self):
        """
        Function to create access functions for the CosmicRay class 
        """
        self.get_keys()
        self.__dict__.update({key: self._fun_factory(key) for key in self.keys})

    def _fun_factory(self, params):
        """
        Helper function to create access functions for the CosmicRay class, explicitily for _create_access_functions 
        """

        def rss_func(val=None):
            return simplified_func(params, val)

        simplified_func = self._combined_access
        return rss_func

    def _combined_access(self, key, val=None):
        """
        Helper function to create access functions for the CosmicRay class, explicitly in _fun_factory
        """
        if val is None:
            return self.__getitem__(key)
        else:
            return self.__setitem__(key, val)

    def get(self, key):
        """
        Getter function to obtain element

        :param key: name of the element
        :type key: string
        :return: value of the element
        """
        return self.__getitem__(key)

    def set(self, key, value):
        """
        Setter function to set values for CosmicRays

        :param key: name of the element
        :type key: string
        :param value: values for all cosmic rays e.g. energy or value the complete set e.g. production version
        """
        self.__setitem__(key, value)

    def get_array(self):
        """Return the numpy array containing the information for all cosmic rays"""
        return self.cosmic_rays

    def get_keys(self):
        """ Function returns all keys like energy, charge, etc, that the class provides"""
        self.keys = list(self.cosmic_rays.dtype.names) + list(self.general_object_store.keys())
        return self.keys

    def load(self, filename):
        """ Loads cosmic rays from a filename

        :param filename: filename from where to load
        :type filename: str
        """
        pkl = True if filename.endswith(".pkl") else False
        if pkl:
            import pickle
            f = open(filename, "rb")
            data = pickle.load(f)
            f.close()
            self.cosmic_rays = data["cosmic_rays"]
            self.general_object_store = data["general_object_store"]
        else:
            self.cosmic_rays = np.load(filename)

    def save(self, filename):
        """
        Save to the given filename

        :param filename: filename where to store the result
        :type filename: str
        """
        pkl = True if filename.endswith(".pkl") else False
        if pkl:
            import pickle
            f = open(filename, "wb")
            pickle.dump({"cosmic_rays": self.cosmic_rays, "general_object_store": self.general_object_store}, f)
            f.close()
        else:
            np.save(filename, self.cosmic_rays)

    def add_cosmic_rays(self, crs):
        """
        Function to add cosmic rays to the already existing set of cosmic rays

        :param crs: numpy array with cosmic rays. The cosmic rays must notc contain all original keys. Missing keys are 
        set to zero. If additional keys are provided, they are ignored 
        """
        try:
            if crs.type == "CosmicRays":
                self.add_cosmic_rays(crs.get_array())
        except AttributeError:
            existing_dtype = self.cosmic_rays.dtype
            cosmic_ray_template = np.zeros(shape=len(crs), dtype=existing_dtype)
            for name in crs.dtype.names:
                cosmic_ray_template[name] = crs[name]
            self.cosmic_rays = np.append(self.cosmic_rays, cosmic_ray_template)
            self._update_attributes()
