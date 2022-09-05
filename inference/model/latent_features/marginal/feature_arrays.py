from typing import Tuple

import torch
from torch import nn, Tensor

from inference.model.latent_features.indexing.key_indexer import KeyIndexer
from inference.model.latent_features.init import IInitializer
from inference.model.latent_features.interface import IFeatureModule
from inference.model.latent_features.marginal.features import FeatureGrid


class IFeatureArray(IFeatureModule):

    def uses_positions(self) -> bool:
        raise NotImplementedError()

    def uses_member(self) -> bool:
        return False

    def uses_time(self) -> bool:
        return False

    def __init__(self, num_keys: int, dimension: int, num_channels: int, debug: bool):
        super(IFeatureArray, self).__init__(dimension, num_channels, debug)
        self._num_keys = num_keys

    def num_keys(self):
        return self._num_keys

    def evaluate(self, positions: Tensor, keys: Tensor) -> Tensor:
        raise NotImplementedError()


class FeatureVectorArray(IFeatureArray):

    def uses_positions(self) -> bool:
        return False

    @classmethod
    def from_initializer(cls, initializer: IInitializer, num_keys: int, num_channels: int, debug=False, device=None, dtype=None):
        data = initializer.get_tensor(num_keys, num_channels, device=device, dtype=dtype)
        return cls(data, debug=debug)

    def __init__(self, data: Tensor, debug=False):
        shape = data.shape
        assert len(shape) == 2
        super(FeatureVectorArray, self).__init__(shape[0], 3, shape[-1], debug)
        self.register_buffer('data', nn.Parameter(data, requires_grad=True))

    def evaluate(self, positions: Tensor, indices: Tensor) -> Tensor:
        return self.data[indices, ...]


class FeatureGridArray(IFeatureArray):

    def uses_positions(self) -> bool:
        return False

    @classmethod
    def from_initializer(cls, initializer, grid_size: Tuple[int, int, int], num_keys: int, num_channels: int, debug=False, device=None, dtype=None):
        data = initializer.get_tensor(num_keys, num_channels, *grid_size, device=device, dtype=dtype)
        return cls(data, debug=debug)

    def __init__(self, data: Tensor, debug=False):
        shape = data.shape
        assert len(shape) == 5
        super(FeatureGridArray, self).__init__(shape[0], 3, shape[1], debug)
        self.key_indexer = KeyIndexer()
        self.feature_mapping = nn.ModuleDict({
            i: FeatureGrid(x, debug=False)
            for i, x in enumerate(data)
        })

    def grid_size(self):
        return tuple(self.data.shape[-3:])

    def evaluate(self, positions: Tensor, indices: Tensor):
        unique_indices, segments = self.key_indexer.query(indices)
        out = torch.zeros(len(positions), self.num_channels())
        for i, segment in zip(unique_indices.tolist(), segments):
            feature = self.feature_mapping[i]
            out[segment] = feature.evaluate(positions[segment])
        return out
