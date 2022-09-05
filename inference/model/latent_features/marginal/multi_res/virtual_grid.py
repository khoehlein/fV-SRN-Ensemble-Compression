from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class VirtualGrid(nn.Module):

    def __init__(self, resolution: Tuple[int, ...], width: Optional[Tensor] = None, offset: Optional[Tensor] = None):
        super(VirtualGrid, self).__init__()
        assert type(resolution) == tuple
        assert np.all([int(s) == s for s in resolution])
        self.register_buffer('resolution', torch.tensor(resolution))
        if width is not None:
            assert type(width) == Tensor
            width = width.reshape(-1)
        else:
            width = torch.ones(self.dimension)
        self.register_buffer('width', width)
        if offset is not None:
            assert type(offset) == Tensor
            offset = offset.reshape(-1)
            assert len(offset) == self.dimension
        else:
            offset = torch.zeros(self.dimension)
        self.register_buffer('offset', offset)
        corner_grid = torch.stack(torch.meshgrid(*[torch.arange(2) for _ in range(self.dimension)]), dim=0)
        self.register_buffer('corner_grid', corner_grid.view(self.dimension, -1))

    @property
    def dimension(self):
        return len(self.resolution)

    def get_corners(self, positions: Tensor, return_residuals=False):
        assert positions.shape[-1] == self.dimension
        y = (positions - self.offset[None, :]) / self.width[None, :]
        y = torch.clip(y, min=0., max=1.)
        n = y * (self.resolution[None, :] - 1)
        bounds = torch.stack([torch.floor(n), torch.ceil(n)], dim=-1)
        corners = [bounds[:, i, self.corner_grid[self.dimension - (i + 1)]] for i in range(self.dimension)]
        corners = torch.stack(corners, dim=1).to(dtype=torch.long)
        if not return_residuals:
            return corners
        residuals = n - bounds[..., 0]
        return corners, residuals


def _test():
    grid = VirtualGrid((4, 4, 4))
    corners = grid.get_corners(torch.tensor([[0.5, 0.5, 0.5]]))
    corners = corners[0].T
    print('Finished')


if __name__ == '__main__':
    _test()
