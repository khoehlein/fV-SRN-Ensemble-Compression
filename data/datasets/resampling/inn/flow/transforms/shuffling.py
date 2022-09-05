from typing import Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from ..interface import InvertibleTransform, FlowDirection


class ChannelShuffle(InvertibleTransform):

    def __init__(self, channels: int, randomize: Optional[bool] = True, seed: Optional[int] = None, label: Optional[str] = None):
        super(ChannelShuffle, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        shuffle_index = np.arange(channels)
        if randomize:
            rng = np.random.Generator(np.random.PCG64(seed))
            rng.shuffle(shuffle_index)
        else:
            shuffle_index = np.concatenate([self.shuffle_index[::2], self.shuffle_index[1::2]], axis=0)
        self.register_buffer('shuffle_index', torch.from_numpy(shuffle_index))
        self.register_buffer('_shuffle_index_inverse', torch.from_numpy(np.argsort(shuffle_index)))

    def forward_transform(self, x: Tensor) -> Tuple[Tensor]:
        out = (x[:, self.shuffle_index],)
        return out

    def reverse_transform(self, x: Tensor) -> Tuple[Tensor]:
        out = (x[:, self._shuffle_index_inverse], )
        return out


class HouseholderShuffle(InvertibleTransform):

    def __init__(self, channels, label: Optional[str] = None):
        super(HouseholderShuffle, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        self.register_parameter('v', nn.Parameter(torch.randn(1, channels, 1, 1), requires_grad=True))

    def _apply_projection(self, x: Tensor) -> Tensor:
        v = self.v / torch.sqrt(torch.sum(self.v ** 2, dim=1, keepdim=True))
        out = (x - 2. * v * torch.sum(x * v, dim=1, keepdim=True),)
        return out

    def forward_transform(self, x: Tensor) -> Tuple[Tensor]:
        return self._apply_projection(x)

    def reverse_transform(self, x: Tensor) -> Tuple[Tensor]:
        return self._apply_projection(x)
