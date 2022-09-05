from torch import Tensor

from .interface import IComponentProcessor


class DirectForward(IComponentProcessor):

    def __init__(self, in_channels: int):
        super(DirectForward, self).__init__(in_channels, in_channels)

    def forward(self, x: Tensor):
        assert x.shape[-1] == self.in_channels()
        return x
