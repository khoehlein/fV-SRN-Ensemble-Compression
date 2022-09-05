import torch
from torch import Tensor


class ISampler(object):

    def __init__(self, dimension: int, device, dtype):
        """
        Base class for everything that generates unconditional random samples in the d-dimensional unit cube [0, 1)**d
        """
        self.dimension = dimension
        if device is None:
            device = torch.device('cpu')
        self.device = device
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype

    def generate_samples(self, num_samples: int) -> Tensor:
        """
        Interface method for querying samples
        """
        raise NotImplementedError()