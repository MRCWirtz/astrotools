# -*- coding: utf-8 -*-
"""
Contains the cosmic rays base class which allows to store arbitrary properties for the cosmic rays and
makes them accesseble via key or getter function.
The second class describes sets of cosmic rays as needed for larger studies.
"""
import matplotlib.pyplot as plt
import numpy as np

from astrotools import healpytools as hpt, skymap

__author__ = 'Martin Urban'

# DTYPE_TEMPLATE = [("pixel", int), ("lon", float), ("lat", float), ("log10e", float), ("charge", float),
#                    ("xmax", float)]
DTYPE_TEMPLATE = [] if np.__version__ >= '1.12' else [("log10e", float)]


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


def plot_eventmap(crs, nside=64, cblabel='log$_{10}$(E / eV)', fontsize=28, opath=None, **kwargs):  # pragma: no cover
    """
    Function to plot a scatter skymap of the cosmic rays

    :param crs: cosmic rays object
    :param cblabel: label for the colorbar
    :param fontsize: Scales the fontsize in the image
    :param opath: Output path for the image, default is None
    :param kwargs:

           - nside: Healpy resolution of the 'pixel' array in the cosmic ray class
           - keywords for function matplotlib.scatter
    """
    if ('lon' in crs.keys()) and ('lat' in crs.keys()):
        vecs = hpt.ang2vec(crs['lon'], crs['lat'])
    elif 'vecs' in crs.keys():
        vecs = crs['vecs']
    else:
        nside = crs['nside'] if 'nside' in crs.keys() else kwargs.pop('nside', 64)
        vecs = hpt.pix2vec(nside, crs['pixel'])

    log10e = crs['log10e']
    idx = np.argsort(log10e)
    skymap.scatter(vecs[:, idx], log10e[idx], cblabel, fontsize, opath=opath, **kwargs)


def plot_healpy_map(crs, opath=None, **kwargs):  # pragma: no cover
    """
    Function to plot a scatter skymap of the cosmic rays

    :param crs: cosmic rays object
    :param opath: Output path for the image, default is None
    :param kwargs:

           - nside: Healpy resolution of the 'pixel' array in the cosmic ray class
           - keywords for function matplotlib.pcolormesh
    """
    nside = crs['nside'] if 'nside' in crs.keys() else kwargs.pop('nside', 64)
    count = np.bincount(crs['pixel'], minlength=hpt.nside2npix(nside))
    skymap.heatmap(count, opath=opath, **kwargs)


def plot_energy_spectrum(crs, xlabel='log$_{10}$(Energy / eV)', ylabel='entries', fontsize=28, bw=0.05,
                         opath=None, **kwargs):  # pragma: no cover
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
    """ Cosmic rays base class meant for inheritance """

    def __init__(self, cosmic_rays):
        self.type = "CosmicRays"
        # needed for the iteration
        self._current_idx = 0  # type: int
        self.general_object_store = {}

        # noinspection PyUnresolvedReferences
        if isinstance(cosmic_rays, str):
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
                raise NotImplementedError("Trying to instantiate the CosmicRays class with a non "
                                          "supported type of cosmic_rays")
        self.ncrs = len(self.cosmic_rays)  # type: int
        self._create_access_functions()

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer, np.ndarray, slice)):
            crs = CosmicRaysBase(self.cosmic_rays[key])
            for k in self.general_object_store.keys():
                to_copy = self.get(k)
                if isinstance(to_copy, (np.ndarray, list)):
                    if len(to_copy) == self.ncrs:
                        to_copy = to_copy[key]
                crs.__setitem__(k, to_copy)
            return crs
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
        else:
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
        Helper function to create access functions for the CosmicRay class, explicitily for _create_access_functions
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
        :type key: string
        :param value: values for all cosmic rays e.g. energy or value the complete set e.g. production version
        """
        self.__setitem__(key, value)

    def get_array(self):
        """Return the numpy array containing the information for all cosmic rays"""
        return self.cosmic_rays

    def keys(self):
        """ Function returns all keys like energy, charge, etc, that the class provides"""
        return list(self.cosmic_rays.dtype.names) + list(self.general_object_store.keys())

    def load(self, filename):
        """ Loads cosmic rays from a filename

        :param filename: filename from where to load
        :type filename: str
        """
        ending = filename.split(".")[-1]
        if ending == "pkl":
            import pickle
            f = open(filename, "rb")
            data = pickle.load(f)
            f.close()
        elif ending == "npy":
            data = np.load(filename).item()
        else:
            filename = filename if filename.endswith(".npz") else filename + ".npz"
            with np.load(filename) as data:
                self.cosmic_rays = data["cosmic_rays"]
                self.general_object_store = data["general_object_store"].item()
        if ending in ["pkl", "npy"]:
            self.cosmic_rays = data["cosmic_rays"]
            self.general_object_store = data["general_object_store"]

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

    def plot_eventmap(self, **kwargs):  # pragma: no cover
        """
        Function to plot a scatter skymap of the cosmic rays

        :param kwargs: additional named arguments.
        """
        plot_eventmap(self, **kwargs)

    def plot_healpy_map(self, **kwargs):  # pragma: no cover
        """
        Function to plot a healpy skymap of the cosmic rays

        :param kwargs: additional named arguments.
        """
        plot_healpy_map(self, **kwargs)

    def plot_energy_spectrum(self, **kwargs):  # pragma: no cover
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

            # Set the shape first as this is required for __setitem__ used by copy from CosmicRaysBase
            CosmicRaysBase.__init__(self, cosmic_rays=ncrs * self.nsets)
            # this number has to be set again as it is overwritten by the init function.
            # It is important to set it before adding the index
            self.type = "CosmicRaysSet"
            self.ncrs = ncrs
            self.shape = (int(self.nsets), int(self.ncrs))
            self.general_object_store["shape"] = self.shape
        elif isinstance(nsets, (list, np.ndarray)):
            self._from_list(nsets)
        else:
            # copy case of a cosmic rays set
            try:
                if nsets.type == self.type:
                    self.general_object_store = {}
                    self.shape = nsets.shape
                    self.copy(nsets)
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
        self._create_access_functions()
        self.ncrs = self.shape[1]
        self.nsets = self.shape[0]

    def _create_access_functions(self):
        CosmicRaysBase._create_access_functions(self)
        if "shape" in self.general_object_store.keys():
            self.shape = self.general_object_store["shape"]

    def _from_list(self, l):
        _nsets = len(l)
        try:
            ncrs_each = np.array([len(elem) for elem in l])
            ncrs = ncrs_each[0]
            types = np.array([elem.type for elem in l])
        except (TypeError, AttributeError):
            raise TypeError("All elements must be of type CosmicRays")
        if not np.all(ncrs_each == ncrs):
            raise ValueError("The number of cosmic rays must be the same in each set")
        if not np.all(types == "CosmicRays"):
            raise TypeError("All elements must be of type CosmicRays")
        keys = [sorted(elem.cosmic_rays.dtype.names) for elem in l]
        joint_keys = np.array(["-".join(elem) for elem in keys])
        gos_keys = [sorted(elem.general_object_store.keys()) for elem in l]
        joint_gos_keys = np.array(["-".join(elem) for elem in gos_keys])
        if not np.all(joint_keys == joint_keys[0]) or not np.all(joint_gos_keys == joint_gos_keys[0]):
            raise AttributeError("All cosmic rays must have the same properties array and general object store")
        self.ncrs = int(ncrs)
        self.nsets = int(_nsets)
        CosmicRaysBase.__init__(self, cosmic_rays=ncrs * self.nsets)
        # reset some elements in the end
        self.type = "CosmicRaysSet"
        self.ncrs = ncrs
        self.shape = (int(self.nsets), int(self.ncrs))
        self.general_object_store["shape"] = self.shape
        for key in keys[0]:
            value = np.array([cr[key] for cr in l])
            self.__setitem__(key, value)
        for key in gos_keys[0]:
            value = np.array([cr[key] for cr in l])
            self.general_object_store[key] = value

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
            for k in self.general_object_store.keys():
                to_copy = self.get(k)
                if isinstance(to_copy, (np.ndarray, list)):
                    if len(to_copy) == self.nsets:
                        to_copy = to_copy[key]
                crs.__setitem__(k, to_copy)
            # The order is important
            crs.ncrs = self.ncrs
            return crs
        elif isinstance(key, (np.ndarray, slice)):
            return self._slice(key)
        elif key in self.general_object_store.keys():
            return self.general_object_store[key]
        else:
            try:
                # casting into int is required to get python3 compatibility
                return np.reshape(self.cosmic_rays[key], self.shape)
            except ValueError as e:
                raise ValueError("The key %s does not exist and the error message was %s" % (key, str(e)))

    def _slice(self, sl):
        if isinstance(sl, slice):
            sl = np.arange(self.nsets)[sl]
        if sl.dtype == bool:
            assert (len(sl) == self.nsets)
            nsets = np.sum(sl)
            sl = np.where(sl)
        elif sl.dtype == int:
            assert (min(sl) >= 0) & (max(sl) < self.nsets)
            nsets = len(sl)
        else:
            raise ValueError("Dtype of ndarray not understood: %s" % (sl.dtype))
        crs = CosmicRaysSets(nsets, self.ncrs)
        for key_copy in self.keys():
            if key_copy not in crs.keys():
                to_copy = self.get(key_copy)
                if isinstance(to_copy, np.ndarray):
                    if len(to_copy) == self.nsets:
                        to_copy = to_copy[sl]
                crs.__setitem__(key_copy, to_copy)
        return crs

    def _update_attributes(self):
        self.ncrs = self.shape[1]
        self.nsets = self.shape[0]

    def plot_eventmap(self, setid=0, **kwargs):  # pragma: no cover
        """
        Function to plot a scatter skymap of the cosmic rays

        :param setid: id of the set which should be plotted
        :param kwargs: additional named arguments.
        """
        # noinspection PyTypeChecker
        crs = self.get(setid)
        plot_eventmap(crs, **kwargs)

    def plot_healpy_map(self, setid=0, **kwargs):  # pragma: no cover
        """
        Function to plot a healpy map of the cosmic ray set

        :param setid: id of the set which should be plotted
        :param kwargs: additional named arguments.
        """
        # noinspection PyTypeChecker
        crs = self.get(setid)
        plot_healpy_map(crs, **kwargs)

    def plot_energy_spectrum(self, setid=0, **kwargs):  # pragma: no cover
        """
        Function to plot the energy spectrum of the cosmic ray set

        :param setid: id of the set which should be plotted
        :param kwargs: additional named arguments.
        """
        # noinspection PyTypeChecker
        crs = self.get(setid)
        plot_energy_spectrum(crs, **kwargs)
