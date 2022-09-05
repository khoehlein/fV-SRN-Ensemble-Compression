import argparse
import math
import os.path
from typing import Dict, Any, Optional

import numpy as np

from data.datasets import DatasetType
from .methods import RandomSampler, HaltonSampler, PlasticSampler


class PositionSampler(object):

    MAX_BATCH_SIZE = 2 ** 14
    METHODS = {
        'plastic': PlasticSampler,
        'halton': HaltonSampler,
        'random': RandomSampler,
    }
    CACHE_PREFIX = '{algorithm}{dimension:d}-'
    CACHE_SUFFIX = '.npy'

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        def add_arguments(group_: argparse._ArgumentGroup, prefix: str, require_or_default):

            def add_argument_with_prefix(arg, default=None, **kwargs):
                if require_or_default and default is None:
                    kwargs.update({'required': True})
                else:
                    kwargs.update({'default': default if require_or_default else None})
                group_.add_argument(prefix + arg, **kwargs)

            add_argument_with_prefix(
                'method', type=str, default='random', choices=['random', 'plastic', 'halton'],
                help="""
                name of sampling algorithm
                """)

        group = parser.add_argument_group('WorldSpacePositionSampling')
        add_arguments(group, '--sampling:', True)  # flags affect both training and validation
        add_arguments(group, f'--sampling:{DatasetType.TRAINING.value}:', False)  # flags affect only training and overwrite previous settings
        add_arguments(group, f'--sampling:{DatasetType.VALIDATION.value}:', False)  # flags affect only validation and overwrite previous settings
        group.add_argument('--sampling:cache', type=str, default=None, help="""
        path to cache directory for position sampler
        """)

    @classmethod
    def from_dict(cls, args: Dict[str, Any], mode: Optional[DatasetType] = DatasetType.TRAINING):

        def get_argument(option):
            argument = args['sampling:' + option] # default argument
            mode_argument = args[f'sampling:{mode.value}:{option}'] # mode argument
            if mode_argument is not None:
                # overwrite default if mode argument is given
                argument = mode_argument
            return argument

        method = get_argument('method')
        cache = args['sampling:cache']
        return cls(method=method, cache=cache, dimension=3)

    def __init__(self, method='random', dimension=3, cache=None):
        self.method = method
        self.sampler = PositionSampler.METHODS[method](dimension)
        self.sample_count = 0
        self.dimension = dimension
        if cache is not None:
            assert os.path.isdir(cache)
        self.cache = cache

    def sample(self, num_samples):
        if self.cache is not None:
            cache_file = self._parse_cache_folder(num_samples)
        else:
            cache_file = None
        if cache_file is not None:
            samples = self._load_samples_from_cache_file(num_samples, cache_file)
        else:
            samples = self._generate_samples(num_samples)
            if self.cache is not None:
                self._store_samples_in_cache(samples)
        return samples

    def _cache_prefix(self):
        return PositionSampler.CACHE_PREFIX.format(algorithm=self.method, dimension=self.dimension)

    def _cache_suffix(self):
        return PositionSampler.CACHE_SUFFIX

    def _parse_cache_folder(self,num_samples):
        prefix = self._cache_prefix()
        suffix = self._cache_suffix()
        best_file = None
        min_num = None
        for f in os.listdir(self.cache):
            if os.path.isfile(os.path.join(self.cache, f)) and f.startswith(prefix) and f.endswith(suffix):
                num = int(f[len(prefix):-len('.npy')])
                if num >= num_samples:
                    if min_num is None or min_num>num:
                        min_num = num
                        best_file = f
        return best_file

    def _load_samples_from_cache_file(self, num_samples, cache_file):
        print("Load",num_samples,"samples from cache file", cache_file)
        content = np.load(os.path.join(self.cache, cache_file), allow_pickle=False)
        return content[:num_samples,:]

    def _generate_samples(self, num_samples):
        batch_size = PositionSampler.MAX_BATCH_SIZE
        num_batches = int(math.ceil(num_samples / batch_size))
        content = np.empty((num_samples, self.dimension), dtype=np.float32)
        for batch in range(num_batches):
            start = batch * batch_size
            end = min(num_samples, start + batch_size)
            indices = np.arange(start, end, dtype=np.int32) + self.sample_count
            content[start:end, :] = self.sampler.sample(indices)
        self.sample_count = self.sample_count + num_samples
        return content

    def _store_samples_in_cache(self, samples):
        num_samples = samples.shape[0]
        output_file = os.path.join(self.cache, self._cache_prefix() + str(num_samples) + self._cache_suffix())
        print("[INFO] Save to cache completed", output_file)
        np.save(output_file, samples)

    def reset(self):
        self.sample_count = 0
        return self
