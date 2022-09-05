from typing import List, Optional, Tuple

from numpy import pi as PI
import torch
from math import log
from torch import nn, Tensor, linalg as la

from torch.nn import functional as F

from .gaussian_latent_space import GaussianLatentSpace
from ..log_likelihood.summation import HierarchicalSum


class JointGaussianLatentSpace(GaussianLatentSpace):
    def __init__(
            self,
            port_shapes: List[Tuple[int, ...]],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
    ):
        super(JointGaussianLatentSpace, self).__init__(port_shapes, dtype=dtype, device=device, sigma=1)
        self.weights = nn.ParameterList([
            nn.Parameter(self._get_initial_weight(shape, device, dtype), requires_grad=True)
            for shape in self.port_shapes
        ])

    def _get_initial_weight(self, shape, device, dtype):
        weight = 10. * torch.eye(shape[0], shape[0], dtype=dtype, device=device)
        weight = weight + torch.randn_like(weight) * 1.e-6
        return weight

    def generate_conditional_sample(self, *x: Tensor, transpose: Optional[bool] = False):
        samples = []
        for i, (shape, condition) in enumerate(zip(self.port_shapes, x)):
            # weight = self._get_weight(i)
            # if transpose:
            #     weight = weight.T
            # mu_cond, sigma_cond = self._compute_conditional_params(condition, weight)
            # sample = self._compute_conditional_sample(mu_cond, sigma_cond)
            # samples.append(sample)
            samples.append(condition)
        return tuple(samples)

    def _get_weight(self, i):
        weight = self.weights[i]
        raw_u = torch.tril(weight, diagonal=-1)
        raw_u = raw_u + torch.eye(*raw_u.shape, device=raw_u.device,dtype=raw_u.dtype)
        u, _ = la.qr(raw_u, mode='complete')
        raw_v = torch.triu(weight, diagonal=1).T
        raw_v = raw_v + torch.eye(*raw_v.shape, device=raw_v.device,dtype=raw_v.dtype)
        v, _ = la.qr(raw_v, mode='complete')
        s = torch.clamp(torch.diagonal(weight), min=-0.999, max=0.999)
        # this was found to be numerically unstable:
        # u, s, vh = la.svd(self.weights[i], full_matrices=False)
        # s = torch.clamp(s, min=-0.999, max=0.999)
        return u @ torch.diag(s) @ v.T

    def _compute_conditional_params(self, x, weight):
        mu_cond = F.conv2d(x, weight[:, :, None, None], bias=None)
        weight_product = weight @ weight.T
        sigma_cond = torch.eye(*weight_product.shape, device=weight_product.device, dtype=weight_product.dtype)
        sigma_cond = sigma_cond - weight_product
        return mu_cond, sigma_cond

    def _compute_conditional_sample(self, mu_cond, sigma_cond):
        eps = torch.randn_like(mu_cond)
        l = la.cholesky(sigma_cond, upper=False)
        deviation = F.conv2d(eps, l[:, :, None, None], bias=None)
        sample = mu_cond + deviation
        return sample

    def compute_conditional_log_likelihood(self, *samples: Tensor, conditions: Optional[List[Tensor]]= None, transpose: Optional[bool] = False) -> Tensor:
        summation = HierarchicalSum()
        for i, (sample, condition) in enumerate(zip(samples, conditions)):
            weight = self._get_weight(i)
            if transpose:
                weight = weight.T
            mu_cond, sigma_cond = self._compute_conditional_params(condition, weight)
            deviation = sample - mu_cond
            shape = tuple(sample.shape)
            l = la.cholesky(sigma_cond)
            exponent = - torch.mean(
                deviation * torch.cholesky_solve(deviation.view(*shape[:2], -1), l, upper=False).view(*shape),
                dim=self._get_summation_dims(shape)
            ) / 2.
            sample_ll = exponent - (la.slogdet(sigma_cond).logabsdet / (2 * shape[0]))
            weight = self.compute_dimensions(shape=shape) / self.dimensions
            summation.add(weight * sample_ll)
        ll_per_dim = summation.value() - log(2. * PI) / 2.
        return ll_per_dim
