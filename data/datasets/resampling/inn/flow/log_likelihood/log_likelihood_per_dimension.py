from typing import Optional, Union, Tuple
from torch import Tensor
import numpy as np

from .summation import HierarchicalSum


class LogLikelihoodPerDimension(object):
    """
    Accumulates log likelihood per data dimension
    """

    def __init__(self, dimensions: Optional[int]=0, ll_per_dim: Optional[Union[int, float, Tensor]]=None):
        self.dimensions = dimensions
        self.summation = HierarchicalSum()
        self.active = False
        if ll_per_dim is not None:
            self.summation.add(ll_per_dim)

    def activate(self):
        self.active = True
        return self

    def deactivate(self):
        self.active = False
        return self

    def add_prior_log_likelihood(self, prior_ll_per_dim: Union[int, float, Tensor], dimensions: int) -> 'LogLikelihoodPerDimension':
        if self.active:
            self.summation.add(self._weighted_log_likelihood(prior_ll_per_dim, dimensions))
        return self

    def add_flow_log_determinant(self, log_abs_det_per_dim: Union[int, float, Tensor], dimensions: int) -> 'LogLikelihoodPerDimension':
        if self.active:
            self.summation.add(self._weighted_log_likelihood(log_abs_det_per_dim, dimensions))
        return self

    def _weighted_log_likelihood(self, ll_per_dim: Union[int, float, Tensor], dimensions: int) -> Union[int, float, Tensor]:
        weight = dimensions / self.dimensions
        return weight * ll_per_dim

    def add_dimensions(self, dimensions: int) -> 'LogLikelihoodPerDimension':
        self.dimensions = self.dimensions + dimensions
        self.summation.reset()
        return self

    def get_value(self) -> Union[int, float, Tensor]:
        return self.summation.value()

    def reset(self) -> None:
        self.summation.reset()

    def flush(self) -> Union[int, float, Tensor]:
        value = self.get_value()
        self.reset()
        return value

    @staticmethod
    def compute_dimensions(x: Optional[Tensor] = None, shape: Optional[Tuple[int, ...]] = None, batch_dim: Optional[int]=0) -> int:
        if x is not None:
            shape = x.shape
        shape = list(shape)
        shape.pop(batch_dim)
        return int(np.prod(shape))
