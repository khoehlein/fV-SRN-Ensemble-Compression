import torch
import torch.nn as nn
from .versionedmodule import VersionedModule


class ConcatenationBlock(VersionedModule):

    __version__ = '1.0'

    def __init__(self):
        super(ConcatenationBlock, self).__init__()

    def forward(self, args):
        return torch.cat(args, dim=1)
