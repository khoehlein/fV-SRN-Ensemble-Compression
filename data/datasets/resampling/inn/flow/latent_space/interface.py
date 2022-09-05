from typing import List, Tuple, Optional, Any, Union
import torch
from torch import Tensor
import torch.nn as nn
import numpy as np


class ILatentSpace(nn.Module):
    def __init__(
            self,
            port_shapes: List[Tuple[int, ...]],
            dtype: Optional[torch.dtype] = None,
            device: Optional[torch.device] = None,
            batch_dim: Optional[int] = 0,
    ):
        super(ILatentSpace, self).__init__()
        self.port_shapes = port_shapes
        self.dtype = dtype
        self.device = device
        self.batch_dim = batch_dim
        self.dimensions = sum([self.compute_dimensions(shape=shape, reduce_batch_dim=False) for shape in port_shapes])

    def generate_sample(self, batch_size: Optional[int] = 1) -> List[Tensor]:
        raise NotImplementedError()

    def compute_log_likelihood(self, *samples: Tensor) -> Tensor:
        raise NotImplementedError()

    def compute_dimensions(self, x: Optional[Tensor] = None, shape: Optional[Tuple[int, ...]] = None, reduce_batch_dim=True) -> int:
        if x is not None:
            shape = x.shape
        shape = list(shape)
        if reduce_batch_dim:
            shape.pop(self.batch_dim)
        return int(np.prod(shape))

    def perturb(self, *code: Any):
        raise NotImplementedError

    def to(
            self,
            device: Optional[Union[int, torch.device]] = ...,
            dtype: Optional[Union[torch.dtype, str]] = ...,
            non_blocking: bool = ...
    ):
        module = super(ILatentSpace, self).to(device=device, dtype=dtype, non_blocking=non_blocking)
        if device is not None:
            module.device = device
        if dtype is not None:
            module.dtype = dtype
        return module
