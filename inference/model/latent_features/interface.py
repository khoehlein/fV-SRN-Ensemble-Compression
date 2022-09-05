from typing import Any

import numpy as np
from torch import nn, Tensor


class IFeatureModule(nn.Module):

    def __init__(self, dimension: int, num_channels: int, debug: bool):
        super(IFeatureModule, self).__init__()
        self.dimension = dimension
        self._num_channels = num_channels
        self._debug = debug

    def num_channels(self):
        return self._num_channels

    def output_channels(self):
        return self.num_channels()

    def set_debug(self, debug):
        self._debug = debug
        return self

    def is_debug(self):
        return self._debug

    def _verify_inputs(self, positions: Tensor, *args: Tensor):
        batch_size = len(positions)
        assert positions.shape[-1] == self.dimension
        assert np.all([batch_size == len(arg) for arg in args])

    def _verify_outputs(self, positions: Tensor, out: Tensor):
        assert len(out) == len(positions)
        assert out.shape[-1] == self.num_channels()

    def evaluate(self, *args) -> Tensor:
        raise NotImplementedError()

    def uses_positions(self) -> bool:
        raise NotImplementedError()

    def uses_member(self) -> bool:
        raise NotImplementedError()

    def uses_time(self) -> bool:
        raise NotImplementedError()


class ILatentFeatures(IFeatureModule):

    def __init__(self, dimension: int, num_channels: int, debug: bool):
        super(ILatentFeatures, self).__init__(dimension, num_channels, debug)

    def evaluate(self, positions: Tensor, time: Tensor, member: Tensor):
        if self.is_debug():
            self._verify_inputs(positions, time, member)
        out = self.forward(positions, time, member)
        if self.is_debug():
            self._verify_outputs(positions, out)
        return out

    def forward(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        raise NotImplementedError()

    def reset_member_features(self, *member_keys: Any) -> 'ILatentFeatures':
        raise NotImplementedError()

    def num_key_times(self) -> int:
        raise NotImplementedError()

    def num_members(self) -> int:
        raise NotImplementedError()

    def uses_positions(self) -> bool:
        raise NotImplementedError()

    def uses_member(self) -> bool:
        raise NotImplementedError()

    def uses_time(self) -> bool:
        raise NotImplementedError()
