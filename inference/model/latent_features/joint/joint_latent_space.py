from typing import List, Any, Optional, Tuple

import pyrenderer
import torch
from torch import Tensor, nn

from inference.model.latent_features.indexing.key_indexer import KeyIndexer
from inference.model.latent_features.init import IInitializer, DefaultInitializer
from inference.model.latent_features.interface import ILatentFeatures
from inference.model.latent_features.marginal.temporal_features import TemporalFeatureVector, TemporalFeatureGrid


class IJointLatentFeatures(ILatentFeatures):
    """
    Module for latent-feature generation for space, time and ensemble coordinates jontly
    """

    def uses_positions(self) -> bool:
        raise NotImplementedError

    def uses_member(self) -> bool:
        return True

    def uses_time(self) -> bool:
        return True

    def export_to_pyrenderer(self, grid_encoding, network: Optional[pyrenderer.SceneNetwork] = None,
                             return_grid_encoding_error=False) -> pyrenderer.SceneNetwork:
        raise RuntimeError('[ERROR] Joint ensemble-time latent features cannot be exported to pyrenderer.')

    def __init__(
            self,
            key_times: List[Any], member_keys: List[Any],
            num_channels: int, initializer: Optional[IInitializer] = None,
            debug=False, dtype=None, device=None
    ):
        super(IJointLatentFeatures, self).__init__(3, num_channels, debug)
        if initializer is None:
            initializer = DefaultInitializer()
        self.initializer = initializer
        self.key_times = key_times
        self.member_index = KeyIndexer()
        self.key_mapping = {}
        self.feature_mapping = nn.ModuleDict({})
        self.device = device
        self.dtype = dtype
        self.reset_member_features(*member_keys)

    def reset_member_features(self, *member_keys: Any) -> 'IJointLatentFeatures':
        raise NotImplementedError()

    def num_key_times(self) -> int:
        return len(self.key_times)

    def num_members(self) -> int:
        return len(self.key_mapping)

    def forward(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        unique_members, segments = self.member_index.query(member)
        out = torch.zeros(len(positions), self.num_channels(), device=self.device, dtype=self.dtype)
        for member, segment in zip(unique_members, segments):
            key = self.key_mapping[member.item()]
            feature = self.feature_mapping[key]
            out[segment] = feature.evaluate(positions[segment], )
        return out


class JointLatentFeatureVector(IJointLatentFeatures):

    def uses_positions(self) -> bool:
        return False

    def reset_member_features(self, *member_keys: Any) -> 'JointLatentFeatureVector':
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleDict({
            i: TemporalFeatureVector(
                self.key_times, self.num_channels(),
                initializer=self.initializer,
                debug=False, dtype=self.dtype, device=self.device
            ) for i in range(len(member_keys))
        })
        return self


class JointLatentFeatureGrid(IJointLatentFeatures):

    def uses_positions(self) -> bool:
        return True

    def __init__(
            self,
            key_times: Tensor, member_keys: List[Any],
            num_channels: int, grid_size: Tuple[int, int, int],
            initializer: Optional[IInitializer] = None,
            debug=False, dtype=None, device=None
    ):
        self._grid_size = grid_size
        super(JointLatentFeatureGrid, self).__init__(
            key_times, member_keys,
            num_channels, initializer=initializer,
            debug=debug, dtype=dtype, device=device
        )

    def grid_size(self) -> Tuple[int, int, int]:
        return self._grid_size

    def reset_member_features(self, *member_keys: Any) -> 'JointLatentFeatureGrid':
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleDict({
            i: TemporalFeatureGrid(
                self.key_times, self.num_channels(), self.grid_size(),
                initializer=self.initializer,
                debug=False, dtype=self.dtype, device=self.device
            ) for i in range(len(member_keys))
        })
        return self
