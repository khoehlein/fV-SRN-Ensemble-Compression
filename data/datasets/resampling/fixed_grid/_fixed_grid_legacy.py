import argparse

import numpy as np
import torch
from torch.nn import functional as F

from inference import IFieldEvaluator


class FixedGridImportanceSampler(object):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('FixedGridImportanceSampler')
        group.add_argument('--importance-sampler:grid-size', type=int, default=64, help="""
        grid resolution for importance-based resampling of the dataset
        """)
        group.add_argument('--importance-sampler:seed', type=int, default=42, help="""
        seed for importance sampling random number generator
        """)
        group.add_argument('--importance-sampler:loss-mode', type=str, default='l1', choices=['l1', 'l2', 'mse'],
                           help="""loss type to compute for loss-importance-based resampling of the dataset (Default: batch size of dataset)""")
        group.add_argument('--importance-sampler:batch-size', type=int, default=None, help="""
                batch size for loss evaluation during importance sampling
                """)
        group.add_argument('--importance-sampler:samples-per-voxel', type=int, default=8, help="""
        number of samples per voxel for loss evaluation in importance-based dataset resampling
        """)
        group.add_argument('--importance-sampler:min-probability', type=float, default=0.01, help="""
        min. probability density for sampling per voxel 
        """)

    def __init__(
            self,
            volume_evaluator, network, grid_size, device,
            batch_size=None, loss_mode='l1',
            seed=42, samples_per_voxel=8, min_probability=0.01
    ):
        self.volume_evaluator = volume_evaluator
        self.network = network
        if not type(grid_size) in [list, tuple]:
            try:
                int_grid_size = int(grid_size)
            except:
                raise ValueError(f'[ERROR] Grid size {grid_size} is not list or tuple and cannot be interpreted as int')
            assert int_grid_size == grid_size, f'[ERROR] Grid size {grid_size} is not list or tuple and cannot be interpreted as int'
            grid_size = tuple([int_grid_size] * 3)
        assert len(grid_size) == 3
        self.grid_size = tuple(grid_size)
        self.device = device
        if batch_size is None:
            batch_size = np.prod(self.grid_size)
        assert int(batch_size) == batch_size, f'[ERROR] Batch size {batch_size} cannot be interpreted as int'
        self.batch_size = int(batch_size)
        self.seed = seed
        self.min_probability = min_probability
        self.samples_per_voxel = samples_per_voxel
        self.sampler_state = 0
        self.loss_mode = loss_mode
        self._loss_function = {'l1': F.l1_loss, 'l2': F.mse_loss, 'mse': F.mse_loss}[loss_mode]

    def restore_defaults(self):
        self.sampler_state = 0
        self.volume_evaluator.restore_defaults()
        return self

    def set_source(self, volume_data, mipmap_level=None):
        self.volume_evaluator.set_source(volume_data, mipmap_level=mipmap_level)
        return self

    def sample(self, num_samples, evaluator):
        loss_grid = self._build_loss_grid(evaluator)
        self._generate_importance_samples(num_samples, loss_grid)
        self.sampler_state = self.sampler_state + 1

    def _generate_importance_samples(self, num_samples, loss_grid):
        max_loss = torch.max(loss_grid).item()
        print(f'Max. Loss ({self.loss_mode}): {max_loss / self.samples_per_voxel}')
        positions, targets, _ = self.volume_evaluator.interpolator.importance_sampling_with_probability_grid(
            num_samples, None, loss_grid, max_loss, self.min_probability, self.seed,
            self.sampler_state, 0., 1.
        )
        return positions, targets

    def _build_loss_grid(self, evaluator: IFieldEvaluator):
        grid_size = self.grid_size
        loss_grid_index = np.mgrid[0:grid_size[0], 0:grid_size[1], 0:grid_size[2]]
        assert loss_grid_index.shape == (3, *grid_size)
        loss_grid_index = np.moveaxis(loss_grid_index, 0, 3)
        loss_grid_numel = np.prod(grid_size)
        num_batches = np.ceil(loss_grid_numel / self.batch_size)
        loss_grid_index_flat = loss_grid_index.view()
        loss_grid_index_flat.shape = (loss_grid_numel, 3)
        with torch.no_grad():
            loss_grid = torch.zeros(grid_size, dtype=torch.float32, device=evaluator.device)
            for j in range(self.samples_per_voxel):
                offsets = np.random.random_sample((loss_grid_numel, 3))
                loss_grid_voxel_size = 1 / np.array(grid_size)[None, :]
                sample_positions = (loss_grid_index_flat + offsets) * loss_grid_voxel_size
                sample_positions = np.array_split(sample_positions, num_batches, axis=0)
                loss_grid_flat = []
                for sample_part in sample_positions:
                    sample_part = torch.from_numpy(sample_part)
                    predictions = evaluator(sample_part)
                    targets = self.volume_evaluator.evaluate(sample_part)
                    loss = self._loss_function(targets, predictions, reduction='none')[:, 0]
                    loss_grid_flat.append(loss)
                loss_grid_flat = torch.cat(loss_grid_flat, dim=0).view(grid_size)
                loss_grid += loss_grid_flat
        return loss_grid
