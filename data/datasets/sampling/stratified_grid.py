from typing import Tuple

import numpy as np
import torch

from .interface import ISampler


class StratifiedGridSampler(ISampler):

    def __init__(self, sampler: ISampler, grid_size: Tuple[int, ...], handle_remainder=False):
        assert len(grid_size) == sampler.dimension
        super(StratifiedGridSampler, self).__init__(len(grid_size), sampler.device, sampler.dtype)
        self.sampler = sampler
        self.grid_size = grid_size
        self.handle_remainder = handle_remainder
        
    def generate_samples(self, num_samples: int):
        grid_size = self.grid_size
        grid_numel = self.grid_numel()
        if not self.handle_remainder:
            assert num_samples % grid_numel == 0
        grid_index = torch.meshgrid(*[torch.arange(s, device=self.device) for s in self.grid_size])
        grid_index = torch.stack(grid_index, dim=-1)
        assert grid_index.shape == (*grid_size, self.dimension)
        grid_index_flat = grid_index.view(-1, self.dimension)
        offsets = self.sampler.generate_samples(num_samples)
        num_samples_per_voxel = num_samples // grid_numel
        voxel_size = 1. / torch.tensor(self.grid_size, device=self.device)
        samples = []
        if num_samples_per_voxel > 0:
            batch_samples = offsets[:(num_samples_per_voxel * grid_numel)]
            batch_samples = batch_samples.view(num_samples_per_voxel, grid_numel, self.dimension)
            batch_samples = (grid_index_flat[None, ...] + batch_samples) *  voxel_size[None, None, ...]
            batch_samples = batch_samples.view(-1, self.sampler.dimension)
            samples.append(batch_samples)
        remainder = num_samples - (num_samples_per_voxel * grid_numel)
        if remainder > 0:
            batch_samples = offsets[-remainder:]
            idx = torch.randperm(grid_numel, device=self.device)[:remainder]
            batch_samples = (grid_index_flat[idx] + batch_samples) * voxel_size[None, ...]
            samples.append(batch_samples)
        if len(samples) == 1:
            return samples[0]
        return torch.cat(samples, dim=0)

    def grid_numel(self):
        return int(np.prod(self.grid_size))
