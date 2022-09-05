import os
from typing import Optional

import h5py
import numpy as np


class _MCCache:
    def __init__(self, settings_file : Optional[str]):
        """
        Creates the cache.
        If the settings_file is None, the cache is disabled, all queries return None.
        :param settings_file: the settings file as reference for the path
        """
        self._cache_filename = None
        self._cache = None
        self._current_tag = None
        self._current_dset = None
        if settings_file is not None:
            self._cache_filename = os.path.abspath(os.path.splitext(settings_file)[0] + "-cache.hdf5")
            if os.path.exists(self._cache_filename):
                self._cache = h5py.File(self._cache_filename, 'r+')
            else:
                self._cache = h5py.File(self._cache_filename, 'w')

    def query(self, actual_tf, actual_timestep, actual_ensemble, num_views, resolution, num_refine):
        """
        Queries the cached image identified by the parameters
        :return: the image as an np.image of shape B,C,H,W or None if not found
        """
        if self._cache is None: return None # cache disabled
        self._current_tag = f"img_{actual_tf}_{actual_timestep}_{actual_ensemble}_{num_views}_{resolution}_{num_refine}"
        if self._current_tag in self._cache:
            # the images are cached!
            return self._cache[self._current_tag][...]
        return None

    def put(self, data: np.ndarray):
        """
        If the cached images was not found, see query(...), put the newly created images into the cache
        :param data:
        :return:
        """
        if self._current_tag is not None:
            # create dataset
            self._cache.create_dataset(self._current_tag, data=data)
            self._current_tag = None

    def close(self):
        if self._cache is not None:
            self._cache.close()
            self._cache = None