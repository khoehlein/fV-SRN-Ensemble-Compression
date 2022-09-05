import numpy as np
from numpy import pi as PI
import torch
from torch import nn, Tensor

from inference.model.input_parameterization.interface import IComponentProcessor


class TrainableFourierFeatures(IComponentProcessor):
    """
    Module for Fourier features with trainable parameter matrix
    """

    def __init__(self, in_channels: int, num_fourier_features: int, dtype=None, device=None):
        super(TrainableFourierFeatures, self).__init__(in_channels, 2 * num_fourier_features)
        num_blocks = int(np.ceil(num_fourier_features / in_channels))
        scales = (2. * PI * torch.pow(2., torch.arange(num_blocks))[:, None]).repeat(1, 3).flatten()
        scales = scales[:num_fourier_features].unsqueeze(0).to(device=device, dtype=dtype)
        self.register_buffer('scales', scales)
        weight = torch.cat([
            torch.eye(in_channels, in_channels, dtype=dtype, device=device)
            for _ in range(num_blocks)
        ], dim=-1)
        self.register_parameter('weight', nn.Parameter(weight[:, :num_fourier_features], requires_grad=True))

    def _fourier_matrix(self):
        return self.scales * self.weight

    def forward(self, x: Tensor):
        assert x.shape[-1] == self.in_channels()
        f2 = torch.matmul(x, self._fourier_matrix())
        return torch.cat([torch.cos(f2), torch.sin(f2)], dim=-1)

    def get_fourier_matrix(self):
        return self._fourier_matrix().data.t()


if __name__ =='__main__':
    module = TrainableFourierFeatures(3, 7)
    a = torch.randn(10, 3)
    out = module(a)
    print(module._fourier_matrix())
    print('Finished')
