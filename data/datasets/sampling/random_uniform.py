import numpy as np
import torch
from torch import Tensor

from .interface import ISampler
from .methods import PlasticSampler, HaltonSampler


class RandomUniformSampler(ISampler):

    def __init__(self, dimension, device=None, dtype=None):
        super(RandomUniformSampler, self).__init__(dimension, device, dtype)

    def generate_samples(self, num_samples: int) -> Tensor:
        return torch.rand(num_samples, self.dimension, device=self.device, dtype=self.dtype)


class _LegacySamplerWrap(ISampler):

    def __init__(self, sampler, dimension, device=None, dtype=None):
        super(_LegacySamplerWrap, self).__init__(dimension, device, dtype)
        self.state = 0
        self._sampler = sampler

    def generate_samples(self, num_samples: int) -> Tensor:
        indices = np.arange(self.state, self.state + num_samples)
        samples = torch.from_numpy(self._sampler.sample(indices)).to(dtype=self.dtype, device=self.device)
        self.state = self.state + num_samples
        return samples


class RandomPlasticSampler(_LegacySamplerWrap):

    def __init__(self, dimension, device=None, dtype=None):
        super(RandomPlasticSampler, self).__init__(PlasticSampler(dimension), dimension, dtype=dtype, device=device)


class RandomHaltonSampler(_LegacySamplerWrap):

    def __init__(self, dimension, device=None, dtype=None):
        super(RandomHaltonSampler, self).__init__(HaltonSampler(dimension), dimension, dtype=dtype, device=device)

