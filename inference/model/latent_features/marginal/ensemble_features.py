from typing import List, Any, Optional, Tuple

import torch
from torch import Tensor, nn

from inference.model.latent_features.marginal.features import FeatureVector, FeatureGrid
from inference.model.latent_features.marginal.multi_res import MultiResolutionFeatures
from inference.model.latent_features.init import IInitializer, DefaultInitializer
from inference.model.latent_features.interface import IFeatureModule


class IEnsembleFeatures(IFeatureModule):

    def uses_member(self) -> bool:
        return True

    def uses_time(self) -> bool:
        return False

    def __init__(
            self,
            member_keys: List[Any], num_channels: int,
            initializer: Optional[IInitializer] = None,
            debug=False, device=None, dtype=None
    ):
        super(IEnsembleFeatures, self).__init__(3, num_channels, debug)
        if initializer is None:
            initializer = DefaultInitializer()
        self.initializer = initializer
        # self.member_index = KeyIndexer()
        self.key_mapping = {}
        self.feature_mapping = nn.ModuleList([])
        self.device = device
        self.dtype = dtype
        self.reset_member_features(*member_keys)

    def evaluate(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        if self.is_debug():
            self._verify_inputs(positions, member)
        out = self.forward(positions, member)
        if self.is_debug():
            self._verify_outputs(positions, out)
        return out

    def forward(self, positions: Tensor, member: Tensor) -> Tensor:
        unique_members = torch.unique(member)
        if len(unique_members) == 1:
            feature = self.feature_mapping[int(unique_members[0].item())]
            return feature.evaluate(positions, None, None)
        out = torch.empty(len(positions), self.num_channels(), device=positions.device, dtype=positions.dtype)
        for umem in unique_members:
            feature = self.feature_mapping[int(umem.item())]
            locations = torch.eq(umem, member)
            out[locations] = feature.evaluate(positions[locations], None, None)
        return out

    def reset_member_features(self, *member_keys: Any) -> 'IEnsembleFeatures':
        raise NotImplementedError()

    def num_members(self) -> int:
        return len(self.key_mapping)

    def uses_positions(self) -> bool:
        raise NotImplementedError()


class EnsembleFeatureVector(IEnsembleFeatures):

    def reset_member_features(self, *member_keys: Any):
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleList([
            FeatureVector.from_initializer(
                self.initializer, self.num_channels(),
                device=self.device, dtype=self.dtype, debug=False
            ) for _ in range(len(member_keys))
        ])
        return self

    def uses_positions(self) -> bool:
        return False


class EnsembleFeatureGrid(IEnsembleFeatures):

    def __init__(
            self,
            member_keys: List[Any], num_channels: int, grid_size: Tuple[int, int, int],
            initializer: Optional[IInitializer] = None,
            debug=False, device=None, dtype=None
    ):
        self._grid_size = grid_size
        super(EnsembleFeatureGrid, self).__init__(
            member_keys, num_channels, initializer=initializer,
            debug=debug, device=device, dtype=dtype
        )

    def grid_size(self):
        return self._grid_size

    def get_grid(self, index:int):
        return self.feature_mapping[index].get_grid()

    def reset_member_features(self, *member_keys: Any):
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleList([
            FeatureGrid.from_initializer(
                self.initializer, self.grid_size(), self.num_channels(),
                device=self.device, dtype=self.dtype, debug=False
            ) for i in range(len(member_keys))
        ])
        return self

    def uses_positions(self) -> bool:
        return True


class EnsembleMultiResolutionFeatures(IEnsembleFeatures):

    def __init__(
            self,
            member_keys: List[Any], num_channels: int,
            coarse_resolution: Tuple[int, int, int], fine_resolution: Tuple[int, int, int],
            num_levels: int, num_nodes: int,
            initializer: Optional[IInitializer] = None,
            debug=False, device=None, dtype=None
    ):
        self._coarse_resolution = coarse_resolution
        self._fine_resolution = fine_resolution
        self._num_levels = num_levels
        self._num_nodes = num_nodes
        super(EnsembleMultiResolutionFeatures, self).__init__(
            member_keys, num_channels, initializer=initializer,
            debug=debug, device=device, dtype=dtype
        )

    def reset_member_features(self, *member_keys: Any):
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleList([
            MultiResolutionFeatures.from_initializer(
                self.initializer, self._coarse_resolution, self._fine_resolution,
                self._num_levels, self._num_nodes, self.num_channels(),
                device=self.device, dtype=self.dtype, debug=False
            ) for i in range(len(member_keys))
        ])
        return self

    def uses_positions(self) -> bool:
        return True


class EnsembleMultiGridFeatures(IEnsembleFeatures):

    def __init__(
            self,
            member_keys: List[Any], num_channels: int, grid_size: Tuple[int, int, int], num_grids: int,
            initializer: Optional[IInitializer] = None,
            debug=False, device=None, dtype=None,
            mixing_mode='normalize'
    ):
        self._grid_size = grid_size
        self._num_grids = num_grids
        self._mixing_mode = mixing_mode
        super(EnsembleMultiGridFeatures, self).__init__(
            member_keys, num_channels, initializer=initializer,
            debug=debug, device=device, dtype=dtype
        )

    def grid_size(self):
        return self._grid_size

    def num_grids(self):
        return self._num_grids

    def reset_member_features(self, *member_keys: Any):
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        mixing_features = self.initializer.get_tensor(
            self.num_members(), self.num_grids(),
            device=self.device, dtype=self.dtype
        )
        self.mixing_features = nn.Parameter(mixing_features, requires_grad=True)
        self.feature_grid = FeatureGrid.from_initializer(
            self.initializer, self.grid_size(), self.num_grids() * self.num_channels(),
            device=self.device, dtype=self.dtype, debug=False
        )
        return self

    def forward(self, positions: Tensor, member: Tensor) -> Tensor:
        features = self.feature_grid.evaluate(positions, None, None)
        mixing_features = self.mixing_features[member.to(dtype=torch.long), ...]
        if not hasattr(self, '_mixing_mode') or self._mixing_mode == 'normalize':
            norm = torch.norm(mixing_features, p=2, dim=-1, keepdim=True)
            mixing_features = mixing_features / norm
        elif self._mixing_mode == 'softmax':
            mixing_features = torch.softmax(mixing_features, dim=-1)
        else:
            raise RuntimeError('[ERROR] Mixing mode must be normalize or softmax.')
        out = torch.bmm(features.view(-1, self.num_channels(), self.num_grids()), mixing_features[..., None])
        return out[..., 0]
