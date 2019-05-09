# -*- coding: utf-8 -*-
"""
Contains a data container which allows to store arbitrary properties and
makes them accesseble via key or getter function.
"""
import numpy as np

__author__ = 'Martin Urban'

DTYPE_TEMPLATE = []


def join_struct_arrays(arrays):
    """
    A function to join a list of numpy named arrays. An alternative (which is much slower) could be:
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
    try:
        sizes = np.array([a.itemsize for a in arrays])
        offsets = np.r_[0, sizes.cumsum()]
        n = len(arrays[0])
        joint = np.empty((n, offsets[-1]), dtype=np.uint8)
        for a, size, offset in zip(arrays, sizes, offsets):
            # as.view() stops returning a view in numpy > 1.16, use repack fields
            # see: https://docs.scipy.org/doc/numpy/user/basics.rec.html
            # from numpy.lib.recfunctions import repack_fields
            # use: repack_fields(a).view(np.uint8)
            joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)
        dtype = sum((a.dtype.descr for a in arrays), [])
        return joint.ravel().view(dtype)
    except TypeError:
        try:
            newdtype = sum((a.dtype.descr for a in arrays), [])
            newrecarray = np.empty(len(arrays[0]), dtype=newdtype)
            for a in arrays:
                for name in a.dtype.names:
                    newrecarray[name] = a[name]
            return newrecarray
        except TypeError as e:
            raise TypeError(str(e))
        except ValueError as e:
            raise ValueError(str(e))


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


# TODO: Do not allow names with leading underscore (if before self.__dict__.update)
class DataContainer(object):
    """ Cosmic rays base class meant for inheritance """

    def __init__(self, cosmic_rays):
        self.type = "CosmicRays"
        # needed for the iteration
        self._current_idx = 0  # type: int
        self.general_object_store = {}

        # noinspection PyUnresolvedReferences
        if cosmic_rays is None:
            self.cosmic_rays = np.empty(0, dtype=DTYPE_TEMPLATE)
        elif isinstance(cosmic_rays, str):
            self.load(cosmic_rays)
        elif isinstance(cosmic_rays, np.ndarray):
            self.cosmic_rays = cosmic_rays
        elif isinstance(cosmic_rays, (int, float, np.integer, np.dtype)):
            if isinstance(cosmic_rays, float):
                if (np.rint(cosmic_rays) != cosmic_rays):
                    raise TypeError("Cosmic rays should not be float type!")
                cosmic_rays = int(cosmic_rays)
            # noinspection PyUnresolvedReferences
            dtype_template = DTYPE_TEMPLATE if isinstance(cosmic_rays, (np.integer, int)) else cosmic_rays
            # noinspection PyUnresolvedReferences
            ncrs = cosmic_rays if isinstance(cosmic_rays, (np.integer, int)) else 0
            cosmic_ray_template = np.zeros(shape=ncrs, dtype=dtype_template)
            self.cosmic_rays = cosmic_ray_template
        else:
            try:
                if isinstance(cosmic_rays, np.void):
                    self.cosmic_rays = np.array([cosmic_rays])
                elif cosmic_rays.type == "CosmicRays":
                    self.copy(cosmic_rays)
            except AttributeError:
                raise NotImplementedError("Trying to instantiate the CosmicRays class with a "
                                          "non supported type of cosmic_rays")
        self.ncrs = len(self.cosmic_rays)  # type: int
        self.shape = (self.ncrs, )
        self._create_access_functions()

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer, np.ndarray, slice)):
            crs = DataContainer(self.cosmic_rays[key])
            for k in self.general_object_store.keys():
                to_copy = self.get(k)
                if isinstance(to_copy, (np.ndarray, list)):
                    if len(to_copy) == self.ncrs:
                        to_copy = to_copy[key]
                crs.__setitem__(k, to_copy)
            return crs
        if key in self.general_object_store.keys():
            return self.general_object_store[key]

        return self.cosmic_rays[key]

    def __setitem__(self, key, value):
        try:
            is_all_crs = len(value) == self.ncrs
            # noinspection PyTypeChecker
            value_shape = len(np.shape(value))
        except TypeError:
            is_all_crs = False
            value_shape = False
        if is_all_crs and value_shape == 1:
            # noinspection PyUnresolvedReferences
            if isinstance(value[0], (float, str, int, np.integer, np.floating)):
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
            except (TypeError, KeyError) as e:
                raise KeyError("This key can not be set and the error message was %s" % str(e))
            except ValueError as e:
                raise ValueError("This value can not be set and the error message was %s" % str(e))
            except Exception as e:
                raise NotImplementedError("An unforeseen error happened: %s" % str(e))

    def __len__(self):
        return int(self.ncrs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """returns next cosmic ray when iterating over all cosmic rays"""
        self._current_idx += 1
        if self._current_idx > self.ncrs:
            self._current_idx = 0
            raise StopIteration
        return self.cosmic_rays[self._current_idx - 1]

    def copy(self, crs):
        """
        Function allows to copy a cosmic ray object to another object

        :param crs: instance of CosmicRays class
        """
        self.cosmic_rays = crs.get_array().copy()
        self._update_attributes()
        for key in crs.keys():
            if key not in self.cosmic_rays.dtype.names:
                self.__setitem__(key, crs[key])

    def _update_attributes(self):
        self.ncrs = len(self.cosmic_rays)

    def _create_access_functions(self):
        """
        Function to create access functions for the CosmicRay class
        """
        self.__dict__.update({key: self._fun_factory(key) for key in self.keys()})

    def _fun_factory(self, params):
        """
        Helper function to create access functions for the CosmicRay class, explicitly for _create_access_functions
        """

        def rss_func(val=None):
            """helper function"""
            return simplified_func(params, val)

        simplified_func = self._combined_access
        return rss_func

    def _combined_access(self, key, val=None):
        """
        Helper function to create access functions for the CosmicRay class, explicitly in _fun_factory
        """
        if val is None:
            return self.__getitem__(key)
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
        :type key: str
        :param value: values for all cosmic rays e.g. energy or value the complete
                      set e.g. production version or arbitrary object
        """
        self.__setitem__(key, value)

    def get_array(self):
        """Return the numpy array containing the information for all cosmic rays"""
        return self.cosmic_rays

    def keys(self):
        """ Function returns all keys like energy, charge, etc, that the class provides"""
        return list(self.cosmic_rays.dtype.names) + list(self.general_object_store.keys())

    def load(self, filename, **kwargs):
        """ Loads cosmic rays from a filename

        :param filename: filename from where to load
        :type filename: str
        :param kwargs: additional keyword arguments passed to numpy / pickle load functions
        """
        ending = filename.split(".")[-1]
        if ending == "pkl":
            import pickle
            f = open(filename, "rb")
            data = pickle.load(f, **kwargs)
            f.close()
        elif ending == "npy":
            data = np.load(filename, allow_pickle=True, **kwargs).item()
        else:
            filename = filename if filename.endswith(".npz") else filename + ".npz"
            with np.load(filename, allow_pickle=True, **kwargs) as data:
                self.cosmic_rays = data["cosmic_rays"]
                self.general_object_store = data["general_object_store"].item()
        if ending in ["pkl", "npy"]:
            self.cosmic_rays = data["cosmic_rays"]
            self.general_object_store = data["general_object_store"]
        if ("shape" in self.general_object_store) and len(self.general_object_store["shape"]) == 2:
            if self.type == "CosmicRays":
                raise AttributeError("Loading a CosmicRaysSets() object with the CosmicRaysBase() class. Use function "
                                     "cosmic_rays.CosmicRaysSets() instead.")

    def save(self, filename):
        """
        Save to the given filename

        :param filename: filename where to store the result
        :type filename: str
        """
        data_dict = {"cosmic_rays": self.cosmic_rays, "general_object_store": self.general_object_store}
        if filename.endswith(".pkl"):
            import pickle
            import sys
            f = open(filename, "wb")
            pickle.dump(data_dict, f, protocol=2 if sys.version_info < (3, 0) else 4)  # fix python 3 pickle dump bug
            f.close()
        elif filename.endswith(".npy"):
            filename = filename if filename.endswith(".npy") else filename + ".npy"
            np.save(filename, data_dict)
        else:
            filename = filename if filename.endswith(".npz") else filename + ".npz"
            np.savez(filename, cosmic_rays=self.cosmic_rays, general_object_store=self.general_object_store)

    def _prepare_readable_output(self, use_keys=None):
        """
        Prepares the ASCII output format

        :param use_keys: list or tuple of keywords that will be used for the saved file
        """
        use_keys = self.keys() if use_keys is None else use_keys
        use_keys_gos = [key for key in self.general_object_store.keys() if key in use_keys]
        use_keys_crs = [key for key in self.cosmic_rays.dtype.names if key in use_keys]

        # build header
        header = ''
        if len(use_keys_gos) > 0:
            header = "General object store information:\n"
            header += "".join(["%s \t %s\n" % (n, self.get(n)) for n in use_keys_gos])
        dtype = self.cosmic_rays.dtype
        header += "\t".join([n for n in use_keys_crs])

        # formatting for displaying decimals
        def t_str(t):
            """ Small function that converts data type to % expressions """
            return "%.6f" if "float" in t else "%s"
        fmt = [t_str(t[0].name) for n, t in dtype.fields.items() if n in use_keys]

        dump = self.cosmic_rays[np.array(use_keys_crs)].copy()    # slices return only a view
        return dump, header, fmt

    def save_readable(self, fname, use_keys=None, **kwargs):
        """
        Saves cosmic ray class as ASCII file with general object store written to header.

        :param fname: file name of the outfile
        :type fname: str
        :param use_keys: list or tuple of keywords that will be used for the saved file
        :param kwargs: additional named keyword arguments passed to numpy.savetxt()
        """
        dump, header, fmt = self._prepare_readable_output(use_keys)
        kwargs.setdefault('header', header)
        kwargs.setdefault('fmt', fmt)
        kwargs.setdefault('delimiter', '\t')
        np.savetxt(fname, dump, **kwargs)
