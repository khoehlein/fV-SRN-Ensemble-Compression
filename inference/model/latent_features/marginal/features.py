from typing import Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inference.model.latent_features.interface import IFeatureModule


class FeatureVector(IFeatureModule):

    def uses_positions(self) -> bool:
        return False

    def uses_member(self) -> bool:
        return False

    def uses_time(self) -> bool:
        return False

    @classmethod
    def from_initializer(
            cls,
            initializer, num_channels: int,
            debug=False, device=None, dtype=None
    ):
        data = initializer.get_tensor(num_channels, device=device, dtype=dtype)
        return cls(data, debug=debug)

    def __init__(self, data: Tensor, debug=False):
        shape = data.shape
        assert len(shape) == 1
        super(FeatureVector, self).__init__(3, shape[0], debug)
        self.register_parameter('data', nn.Parameter(data, requires_grad=True))

    def evaluate(self, positions: Tensor, time: Tensor, member: Tensor):
        return torch.tile(self.data[None, :], (len(positions), 1))


class FeatureGrid(IFeatureModule):

    def uses_positions(self) -> bool:
        return True

    def uses_member(self) -> bool:
        return False

    def uses_time(self) -> bool:
        return False

    @classmethod
    def from_initializer(
            cls,
            initializer, grid_size: Tuple[int, int, int], num_channels: int,
            debug=False, device=None, dtype=None
    ):
        data = initializer.get_tensor(num_channels, *grid_size, device=device, dtype=dtype)
        return cls(data, debug=debug)

    def __init__(self, data: Tensor, debug=False):
        shape = data.shape
        assert len(shape) == 4
        super(FeatureGrid, self).__init__(3, shape[0], debug)
        self.register_parameter('data', nn.Parameter(data, requires_grad=True))

    def grid_size(self):
        return tuple(self.data.shape[-3:])

    def evaluate(self, positions: Tensor, time: Tensor, member: Tensor):
        grid = 2. * positions - 1.
        grid = grid.view(*[1 for _ in self.grid_size()], *positions.shape)
        samples = F.grid_sample(self.data[None, ...], grid, mode='bilinear', align_corners=False, padding_mode='border')
        out = samples.view(self.num_channels(), len(positions)).T
        return out

    def get_grid(self):
        return self.data

    def num_channels(self):
        return self.data.shape[0]
