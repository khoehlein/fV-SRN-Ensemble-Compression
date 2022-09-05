import torch
from .versionedmodule import VersionedModule


class AdditiveGaussianNoise(VersionedModule):

    __version__ = '1.0'

    def __init__(self, noise_amplitude=1.0):
        super(AdditiveGaussianNoise, self).__init__()
        self.amplitude = noise_amplitude

    def forward(self, x):
        noise = torch.randn_like(x)
        return x + self.amplitude * noise
