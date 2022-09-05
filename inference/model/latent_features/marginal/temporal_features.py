from typing import Optional, Tuple, List, Any

from torch import Tensor

from inference.model.latent_features.indexing.time_indexer import TimeIndexer
from inference.model.latent_features.init import IInitializer, DefaultInitializer
from inference.model.latent_features.interface import IFeatureModule
from inference.model.latent_features.marginal.feature_arrays import FeatureVectorArray, FeatureGridArray


class ITemporalFeatures(IFeatureModule):

    def uses_member(self) -> bool:
        return False

    def uses_time(self) -> bool:
        return True

    def __init__(self, key_times: List[Any], dimension: int, num_channels: int, debug: bool, device):
        super(ITemporalFeatures, self).__init__(dimension, num_channels, debug)
        self.key_time_index = TimeIndexer(key_times, device=device)

    def num_key_times(self):
        return self.key_time_index.num_key_times()

    def evaluate(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        if self.is_debug():
            self._verify_inputs(positions, time)
        out = self.forward(positions, time)
        if self.is_debug():
            self._verify_outputs(positions, out)
        return out

    def forward(self, positions: Tensor, time: Tensor) -> Tensor:
        raise NotImplementedError()

    def uses_positions(self):
        raise NotImplementedError()


class TemporalFeatureVector(ITemporalFeatures):

    def __init__(
            self,
            key_times: List[Any],
            num_channels: int, initializer: Optional[IInitializer] = None,
            debug: Optional[bool] = False,
            dtype=None, device=None
    ):
        super(TemporalFeatureVector, self).__init__(key_times, 3, num_channels, debug, device)
        if initializer is None:
            initializer = DefaultInitializer()
        self.key_features = FeatureVectorArray.from_initializer(
            initializer, self.num_channels(), self.num_key_times(),
            device=device, dtype=dtype, debug=False
        )

    def forward(self, positions: Tensor, time: Tensor) -> Tensor:
        lower, upper, fraction = self.key_time_index.query(time)
        fraction = fraction[:, None]
        features_lower = self.key_features.evaluate(positions, lower)
        features_upper = self.key_features.evaluate(positions, upper)
        features = (1. - fraction) * features_lower + fraction * features_upper
        return features

    def uses_positions(self):
        return False


class TemporalFeatureGrid(ITemporalFeatures):

    def __init__(
            self,
            key_times: List[Any],
            num_channels: int, grid_size: Tuple[int, int, int],
            initializer: Optional[IInitializer] = None,
            debug: Optional[bool] = False, dtype=None, device=None
    ):
        super(TemporalFeatureGrid, self).__init__(key_times, 3, num_channels, debug, device)
        if initializer is None:
            initializer = DefaultInitializer()
        self._grid_size = grid_size
        self.key_features = FeatureGridArray.from_initializer(
            initializer, grid_size, self.num_key_times(), self.num_channels(),
            debug=False, device=device, dtype=dtype
        )

    def grid_size(self):
        return self._grid_size

    def forward(self, positions: Tensor, time: Tensor) -> Tensor:
        lower, upper, fraction = self.key_time_index.query(time)
        features_lower = self.key_features.evaluate(positions, lower)
        features_upper = self.key_features.evaluate(positions, upper)
        features = features_lower + fraction[:, None] * features_upper
        return features

    def uses_positions(self):
        return True
