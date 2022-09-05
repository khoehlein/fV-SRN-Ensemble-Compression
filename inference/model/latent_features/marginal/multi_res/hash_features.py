from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from torch.nn import functional as F

from inference.model.latent_features.marginal.multi_res.virtual_grid import VirtualGrid
from inference.model.latent_features.init import DefaultInitializer
from inference.model.latent_features.interface import IFeatureModule


class HashedFeatureGrid(IFeatureModule):
    """
    Hashed features by Müller et al. (2022): http://arxiv.org/abs/2201.05989
    """

    def evaluate(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        batch_size = positions.shape[0]
        corners, residuals = self.grid.get_corners(positions, return_residuals=True)
        hashes = self._compute_hashes(corners)
        features = self.features[:, hashes.view(-1)]
        features = features.view(self.num_channels(), batch_size, *[2 for _ in range(self.grid.dimension)])
        features = torch.transpose(features, 1, 0)
        grid = (2. * residuals - 1.)[:, None, None, None, :]
        out = F.grid_sample(features, grid, align_corners=True, padding_mode='border', mode='bilinear')
        out = out.view(batch_size, self.num_channels())
        return out

    def _compute_hashes(self, corners: Tensor):
        corners = corners * self.hash_primes[None, :, None]
        hashes = corners[:, 0, :]
        for i in range(1, self.grid.dimension):
            hashes = torch.bitwise_xor(hashes, corners[:, i, :])
        return hashes % self.num_nodes

    def uses_positions(self) -> bool:
        return True

    def uses_member(self) -> bool:
        return False

    def uses_time(self) -> bool:
        return False

    def __init__(self, data: Tensor, grid: VirtualGrid, hash_primes: Tuple[int, ...] = None, debug=False):
        num_channels, num_nodes = data.shape
        super(HashedFeatureGrid, self).__init__(grid.dimension, num_channels, debug)
        self.grid = grid
        self.register_parameter('features', nn.Parameter(data, requires_grad=True))
        if hash_primes is not None:
            assert len(hash_primes) == grid.dimension
            hash_primes = torch.tensor(hash_primes)
        else:
            # default configuration from http://arxiv.org/abs/2201.05989
            hash_primes = torch.tensor([1, 2654435761, 805459861])
        self.register_buffer('hash_primes', hash_primes.to(dtype=torch.long))
        self.num_nodes = num_nodes

    @classmethod
    def from_initializer(
            cls,
            initializer, grid_size: Tuple[int, int, int], num_nodes: int, num_channels: int,
            grid_width=None, grid_offset=None, hash_primes=None,
            debug=False, device=None, dtype=None
    ):
        grid = VirtualGrid(grid_size, width=grid_width, offset=grid_offset)
        data = initializer.get_tensor(num_channels, num_nodes, device=device, dtype=dtype)
        return cls(data, grid, hash_primes=hash_primes, debug=debug)


def _test():
    feature_grid = HashedFeatureGrid.from_initializer(
        DefaultInitializer(), (2, 4, 6), 100, 1
    )
    positions = torch.ones(20, 3) * 0.5
    positions[:, 0] = torch.linspace(0, 1, 20)
    features = feature_grid.evaluate(positions, None, None)
    features = features[:, 0].data.cpu().numpy()
    plt.scatter(np.arange(20), features)
    plt.show()
    print('Finished!!!')


if __name__ == '__main__':
    _test()
