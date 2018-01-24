# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from astrotools import healpytools as hpt, skymap

__author__ = 'Martin Urban'

# _dtype_template = [("pixel", int), ("lon", float), ("lat", float), ("log10e", float), ("charge", float),
#                    ("xmax", float)]
_dtype_template = [] if np.__version__ >= '1.12' else [("log10e", float)]


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
    try:
        sizes = np.array([a.itemsize for a in arrays])
        offsets = np.r_[0, sizes.cumsum()]
        n = len(arrays[0])
        joint = np.empty((n, offsets[-1]), dtype=np.uint8)
        for a, size, offset in zip(arrays, sizes, offsets):
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


def plot_eventmap(crs, nside=64, cblabel='log$_{10}$(Energy / eV)', fontsize=28, opath=None, **kwargs):
    """
    Function to plot a scatter skymap of the cosmic rays
    :param crs: cosmic rays object

    :param nside: Healpy resolution of the 'pixel' array in the cosmic ray class.
    :param cblabel: label for the colorbar
    :param fontsize: Scales the fontsize in the image.
    :param opath: Output path for the image, default is None 
    """
    pixel = crs['pixel']
    log10e = crs['log10e']
    skymap.scatter(hpt.rand_vec_in_pix(nside, pixel), log10e, cblabel, fontsize, opath=opath, **kwargs)


def plot_healpy_map(crs, nside=64, opath=None, **kwargs):
    """
    Function to plot a scatter skymap of the cosmic rays
    :param crs: cosmic rays object

    :param nside: Healpy resolution of the 'pixel' array in the cosmic ray class.
    :param opath: Output path for the image, default is None
    """
    pixel = crs['pixel']
    count = np.bincount(pixel, minlength=hpt.nside2npix(nside))
    skymap.skymap(count, opath=opath, **kwargs)


def plot_energy_spectrum(crs, xlabel='log$_{10}$(Energy / eV)', ylabel='entries', fontsize=28, bw=0.05,
                         opath=None, **kwargs):
    """
    Function to plot the energy spectrum of the cosmic ray set
    :param crs: cosmic rays object
    :param xlabel: label for the x-axis
    :param ylabel: label for the y-axis
    :param fontsize: Scales the fontsize in the image.
    :param bw: bin width for the histogram
    :param opath: Output path for the image, default is None 
    """
    log10e = crs['log10e']
    bins = np.arange(17., 20.6, bw)
    bins = bins[(bins >= np.min(log10e) - 0.1) & (bins <= np.max(log10e) + 0.1)]
    plt.hist(log10e, bins=bins, histtype='step', fill=None, color='k', **kwargs)
    plt.xticks(fontsize=fontsize - 4)
    plt.yticks(fontsize=fontsize - 4)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    if opath is not None:
        plt.savefig(opath, bbox_inches='tight')
        plt.clf()


# TODO: Do not allow names with leading underscore (if before self.__dict__.update)
class CosmicRaysBase:
    def __init__(self, cosmic_rays=None):
        self.type = "CosmicRays"
        # needed for the iteration
        self._current_idx = 0  # type: int
        self.general_object_store = {}
        if cosmic_rays is None:
            raise NotImplementedError(
                "Either the number of cosmic rays has to be set or the numpy array with cosmic rays has to be given"
                "or a filename to load cosmic rays from has to be given")
        # noinspection PyUnresolvedReferences
        if isinstance(cosmic_rays, str):
            self.load(cosmic_rays)
        elif isinstance(cosmic_rays, np.ndarray):
            self.cosmic_rays = cosmic_rays
        elif isinstance(cosmic_rays, (int, float, np.integer, np.dtype)):
            if isinstance(cosmic_rays, float):
                print(UserWarning("cosmic rays should not be float type, casting it to an int"))
                cosmic_rays = int(cosmic_rays)
            # noinspection PyUnresolvedReferences
            dtype_template = _dtype_template if isinstance(cosmic_rays, (np.integer, int)) else cosmic_rays
            # noinspection PyUnresolvedReferences
            ncrs = cosmic_rays if isinstance(cosmic_rays, (np.integer, int)) else 0
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
        # noinspection PyUnresolvedReferences
        if isinstance(key, (int, np.integer)) or isinstance(key, (list, np.ndarray)):
            return self.cosmic_rays[key]
        if key in self.general_object_store.keys():
            return self.general_object_store[key]
        else:
            try:
                return self.cosmic_rays[key]
            except ValueError as e:
                raise ValueError("The key '%s' does not exist and the error message was %s" % (key, str(e)))

    def __setitem__(self, key, value):
        if key in self.cosmic_rays.dtype.names:
            self.cosmic_rays[key] = value
            return
        try:
            is_all_crs = len(value) == self.ncrs
            # noinspection PyTypeChecker
            value_shape = len(np.shape(value))
        except TypeError:
            is_all_crs = False
            value_shape = False
        if is_all_crs and value_shape <= 1:
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
            except KeyError as e:
                raise KeyError("This key can not be set and the error message was %s" % str(e))
            except ValueError as e:
                raise KeyError("This value can not be set and the error message was %s" % str(e))
            except Exception as e:
                raise NotImplementedError("An unforeseen error happened: %s" % str(e))
        self.keys = self.get_keys()

    def __len__(self):
        return int(self.ncrs)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        self._current_idx += 1
        if self._current_idx >= self.ncrs:
            self._current_idx = 0
            raise StopIteration
        else:
            return self.cosmic_rays[self._current_idx - 1]

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
        if filename.endswith(".pkl"):
            import pickle
            f = open(filename, "rb")
            data = pickle.load(f)
            f.close()
        elif filename.endswith(".npy"):
            filename = filename if filename.endswith(".npy") else filename + ".npy"
            data = np.load(filename).item()
        else:
            filename = filename if filename.endswith(".npz") else filename + ".npz"
            data = np.load(filename)
        self.cosmic_rays = data["cosmic_rays"]
        self.general_object_store = self.general_object_store = data["general_object_store"] if filename[-4:] in [
            ".npy", ".pkl"] else data["general_object_store"].item()

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

    def plot_eventmap(self, **kwargs):
        """
        Function to plot a scatter skymap of the cosmic rays

        :param kwargs: additional named arguments.
        """
        plot_eventmap(self.cosmic_rays, **kwargs)

    def plot_healpy_map(self, **kwargs):
        """
        Function to plot a healpy skymap of the cosmic rays

        :param kwargs: additional named arguments.
        """
        plot_healpy_map(self.cosmic_rays, **kwargs)

    def plot_energy_spectrum(self, **kwargs):
        """
        Function to plot the energy spectrum of the cosmic ray set

        :param kwargs: additional named arguments. 
        """
        plot_energy_spectrum(self.cosmic_rays, **kwargs)


class CosmicRaysSets(CosmicRaysBase):
    """Set of cosmic rays """

    def __init__(self, nsets, ncrs=None):
        self.type = "CosmicRaysSet"
        if nsets is None:
            raise NotImplementedError(
                "Either the number of cosmic rays has to be set or the numpy array with cosmic rays has to be given"
                "or a filename to load cosmic rays from has to be given")
        # noinspection PyUnresolvedReferences
        if isinstance(nsets, str):
            self.load(nsets)
        elif isinstance(nsets, (tuple, float, int, np.integer)):
            self.nsets = nsets[0] if isinstance(nsets, tuple) else nsets
            ncrs = nsets[1] if isinstance(nsets, tuple) else ncrs

            # Set the shape first as this is required for __setitem__ used by __copy__ from CosmicRaysBase
            CosmicRaysBase.__init__(self, cosmic_rays=ncrs * self.nsets)
            # this number has to be set again as it is overwritten by the init function.
            # It is important to set it before adding the index
            self.type = "CosmicRaysSet"
            self.ncrs = ncrs
            self.shape = (int(self.nsets), int(self.ncrs))
            self.general_object_store["shape"] = self.shape
        else:
            # copy case of a cosmic rays set
            try:
                if nsets.type == self.type:
                    self.general_object_store = {}
                    self.shape = nsets.shape
                    self.__copy__(nsets)
                    self.keys = self.get_keys()
                    self._create_access_functions()
                    # _create_access_functions and the __setitem__ function from the CosmicRaysBase overwrite self.shape
                    self.shape = nsets.shape
            except AttributeError as e:
                raise AttributeError(str(e))
                # raise NotImplementedError("Trying to instantiate the CosmicRaysSets class with a non "
                #                           "supported type of cosmic_rays")

    def load(self, filename):
        """ Loads cosmic rays from a filename

        :param filename: filename from where to load
        :type filename: str
        """
        CosmicRaysBase.load(self, filename)
        self.shape = self.general_object_store["shape"]
        self.ncrs = self.shape[1]
        self.nsets = self.shape[0]

    def __setitem__(self, key, value):
        # casting into int is required to get python3 compatibility
        v = value.reshape(int(self.nsets * self.ncrs)) if np.shape(value) == self.shape else value
        # to avoid the overwriting we use this hack
        self.ncrs = self.ncrs * self.nsets
        CosmicRaysBase.__setitem__(self, key, v)
        # this number has to be set again as it is overwritten by the init function
        self.ncrs /= int(self.nsets)

    def __getitem__(self, key):
        # noinspection PyUnresolvedReferences
        if isinstance(key, (int, np.integer)):
            crs = CosmicRaysBase(self.cosmic_rays.dtype)
            idx_begin = int(key * self.ncrs)
            idx_end = int((key + 1) * self.ncrs)
            crs.cosmic_rays = self.cosmic_rays[idx_begin:idx_end]
            crs.general_object_store = self.general_object_store
            # crs.general_object_store["_parent"] = self
            # crs.general_object_store["_slice"] = key
            # The order is important
            crs.ncrs = self.ncrs
            return crs
        elif isinstance(key, slice):
            raise NotImplementedError("Getting a slice from a set is currently not supported")
        elif key in self.general_object_store.keys():
            return self.general_object_store[key]
        else:
            try:
                # casting into int is required to get python3 compatibility
                return np.reshape(self.cosmic_rays[key], self.shape)
            except ValueError as e:
                raise ValueError("The key %s does not exist and the error message was %s" % (key, str(e)))

    def _update_attributes(self):
        self.ncrs = self.shape[1]
        self.nsets = self.shape[0]

    def plot_eventmap(self, setid=0, **kwargs):
        """
        Function to plot a scatter skymap of the cosmic rays

        :param setid: id of the set which should be plotted
        :param kwargs: additional named arguments. 
        """
        # noinspection PyTypeChecker
        crs = self.get(setid)
        plot_eventmap(crs, **kwargs)

    def plot_healpy_map(self, setid=0, **kwargs):
        """
        Function to plot a healpy map of the cosmic ray set

        :param setid: id of the set which should be plotted
        :param kwargs: additional named arguments.
        """
        # noinspection PyTypeChecker
        crs = self.get(setid)
        plot_healpy_map(crs, **kwargs)

    def plot_energy_spectrum(self, setid=0, **kwargs):
        """
        Function to plot the energy spectrum of the cosmic ray set

        :param setid: id of the set which should be plotted
        :param kwargs: additional named arguments. 
        """
        # noinspection PyTypeChecker
        crs = self.get(setid)
        plot_energy_spectrum(crs, **kwargs)
