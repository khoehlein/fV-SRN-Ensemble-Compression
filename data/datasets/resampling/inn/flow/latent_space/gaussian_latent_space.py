from typing import Optional, List, Tuple
from numpy import pi as PI
from math import log, sqrt
import torch
from torch import Tensor
from .interface import ILatentSpace
from ..log_likelihood.summation import HierarchicalSum


class GaussianLatentSpace(ILatentSpace):

    def __init__(
            self,
            port_shapes: List[Tuple[int, ...]],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            sigma: Optional[float] = 1.
    ):
        super(GaussianLatentSpace, self).__init__(port_shapes, dtype=dtype, device=device)
        self.sigma = sigma

    def generate_sample(self, batch_size: Optional[int] = 1) -> List[Tensor]:
        return [self._get_sample_tensor(shape, batch_size) for shape in self.port_shapes]

    def _get_sample_tensor(self, shape: Tuple[int, ...], batch_size: int) -> Tensor:
        raw_sample = torch.randn(
            *self._get_sample_shape(shape, batch_size),
            dtype=self.dtype, device=self.device
        )
        sample = self.sigma * raw_sample
        return sample

    def _get_sample_shape(self, shape, batch_size):
        sample_shape = list(shape)
        sample_shape.insert(self.batch_dim, batch_size)
        return tuple(sample_shape)

    def compute_log_likelihood(self, *samples: Tensor) -> Tensor:
        summation = HierarchicalSum()
        for sample_tensor in samples:
            sample_ll = - torch.pow(torch.abs(sample_tensor), 2.) / (2. * self.sigma ** 2)
            sample_ll = torch.mean(sample_ll, dim=self._get_summation_dims(tuple(sample_tensor.shape)))
            weight = self.compute_dimensions(shape=sample_tensor.shape) / self.dimensions
            summation.add(weight * sample_ll)
        ll_per_dim = summation.value() - log(2. * PI * self.sigma ** 2.) / 2.
        return ll_per_dim

    def _get_summation_dims(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        dims = list(range(len(shape)))
        dims.pop(self.batch_dim)
        dims = tuple(dims)
        return dims

    def perturb(self, *code: Tensor, num_samples: Optional[int] = 1, sigma: Optional[float] = 1.):
        if num_samples > 1:
            code = [torch.repeat_interleave(c, num_samples, dim=self.batch_dim) for c in code]
        perturbed = [(c + sigma * torch.randn_like(c)) / sqrt(self.sigma ** 2 + sigma ** 2) for c in code]
        return perturbed
