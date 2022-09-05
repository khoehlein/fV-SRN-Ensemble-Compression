import torch
from utils.misc.pytorch import VersionedModule


class Clamp(VersionedModule):

    __version__ = '1.0'

    def __init__(self, lower=None, upper=None):
        super(Clamp, self).__init__()
        self.lower = lower
        self.upper = upper

    def forward(self, x):
        return torch.clamp(x, min=self.lower, max=self.upper)