import torch
import torch.nn as nn
from .versionedmodule import VersionedModule


class CrossConnection(VersionedModule):

    __version__ = '1.0'

    def __init__(self, inner_module):
        super(CrossConnection, self).__init__()
        assert isinstance(inner_module, nn.Module)
        self.inner_module = inner_module

    def forward(self, x):
        return torch.cat((x, self.inner_module(x)), dim=1)
