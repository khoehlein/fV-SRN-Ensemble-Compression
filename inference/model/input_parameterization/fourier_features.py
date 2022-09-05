from typing import Optional

import numpy as np
from numpy import pi as PI
import torch
from torch import Tensor

from .interface import IComponentProcessor


class IFourierFeatures(IComponentProcessor):
    """
    Base class for Fourier feature methods.
    All sub-classes should use Fourier matrices B, which are pre-multiplied with 2 * PI!
    """

    def __init__(self, fourier_matrix: Tensor):
        in_channels, half_out_channels = fourier_matrix.shape
        super(IFourierFeatures, self).__init__(in_channels, 2 * half_out_channels)
        self.register_buffer('fourier_matrix', fourier_matrix)

    def forward(self, x: Tensor):
        assert x.shape[-1] == self.in_channels()
        f2 = torch.matmul(x, self.fourier_matrix)
        return torch.cat([torch.cos(f2), torch.sin(f2)], dim=-1)

    def get_fourier_matrix(self):
        return self.fourier_matrix.t()


class RandomFourierFeatures(IFourierFeatures):

    def __init__(self, in_channels: int, num_fourier_features: int, std: Optional[float] = 1.):
        B = 2. * PI * torch.normal(0, std, (num_fourier_features, in_channels))
        super(RandomFourierFeatures, self).__init__(B.t())


class NerfFourierFeatures(IFourierFeatures):

    def __init__(self, in_channels: int, num_fourier_features: int):
        num_blocks = int(np.ceil(num_fourier_features / in_channels))
        B = [2 ** i * torch.eye(in_channels, in_channels) for i in range(num_blocks)]
        B = 2. * PI * torch.cat(B, dim=0)[:num_fourier_features, :]
        super(NerfFourierFeatures, self).__init__(B.t())
