import torch
from torch import Tensor, nn


class CoordinateBox(nn.Module):

    def __init__(self, dimension: int, bounds: Tensor):
        super(CoordinateBox, self).__init__()
        assert dimension > 0, '[ERROR] Dimensions must be positive.'
        self.dimension = dimension
        assert bounds.shape == (2, dimension), \
            f'[ERROR] Expecting bounds tensor to have shape (2, {dimension}). Got {tuple(bounds.shape)} instead'
        assert torch.all(bounds[0] < bounds[1]), \
            f'[ERROR] Expecting lower bounds to be strictly smaller than upperbounds. Got lower bounds {tuple(bounds[0].tolist())} and upper bounds {tuple(bounds[1].tolist())} instead.'
        self.register_buffer('bounds', bounds)

    @property
    def device(self):
        return self.bounds.device

    @property
    def dtype(self):
        return self.bounds.dtype

    def center(self, keepdim=True):
        return torch.mean(self.bounds, dim=0, keepdim=keepdim)

    def lower_bounds(self, keepdim=True):
        bounds = self.bounds[0]
        if keepdim:
            return bounds[None, ...]
        return bounds

    def upper_bounds(self, keepdim=True):
        bounds = self.bounds[1]
        if keepdim:
            return bounds[None, ...]
        return bounds

    def size(self, keepdim=True):
        box_size = torch.diff(self.bounds, dim=0)
        if not keepdim:
            box_size = box_size[0]
        return box_size

    def volume(self):
        return torch.prod(self.size(keepdim=False)).item()

    def max_aspect(self):
        size = self.size(keepdim=False)
        return (torch.max(size) / torch.min(size)).item()

    def contains(self, coordinates: Tensor, include_boundary=True):
        assert coordinates.shape[-1] == self.dimension, \
            f'[ERROR] Expecting last axis of coordinate array to have dimension {self.dimension}. Got {coordinates.shape[-1]} instead.'
        leading_axes = len(coordinates.shape) - 1
        bounds_shape = [1 for _ in range(leading_axes)]
        lower_bounds = self.lower_bounds(keepdim=False).view(*bounds_shape, self.dimension)
        upper_bounds = self.upper_bounds(keepdim=False).view(*bounds_shape, self.dimension)
        if include_boundary:
            satisfies_lower_bound = lower_bounds <= coordinates
            satisfies_upper_bound = coordinates <= upper_bounds
        else:
            satisfies_lower_bound = lower_bounds < coordinates
            satisfies_upper_bound = coordinates < upper_bounds
        return torch.logical_and(torch.all(satisfies_lower_bound, dim=-1), torch.all(satisfies_upper_bound, dim=-1))

    def rescale(self, coordinates: Tensor):
        assert coordinates.shape[-1] == self.dimension, \
            f'[ERROR] Expecting last axis of coordinate array to have dimension {self.dimension}. Got {coordinates.shape[-1]} instead.'
        leading_axes = len(coordinates.shape) - 1
        bounds_shape = [1 for _ in range(leading_axes)]
        lower_bounds = self.lower_bounds(keepdim=False).view(*bounds_shape, self.dimension)
        upper_bounds = self.upper_bounds(keepdim=False).view(*bounds_shape, self.dimension)
        delta = upper_bounds - lower_bounds
        return coordinates * delta + lower_bounds


class UnitCube(CoordinateBox):

    def __init__(self, dimension: int, device=None, dtype=None):
        assert dimension > 0,'[ERROR] Dimensions must be positive.'
        bounds = torch.tensor([[0., 1.]] * dimension, device=device, dtype=dtype).T
        super(UnitCube, self).__init__(dimension, bounds)
