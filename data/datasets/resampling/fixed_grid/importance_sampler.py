import argparse
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
import torch.nn.functional as F

from inference import IFieldEvaluator
from data.datasets.resampling import IImportanceSampler, CoordinateBox
from data.datasets.sampling import ISampler, RandomUniformSampler, StratifiedGridSampler


class FixedGridImportanceSampler(IImportanceSampler):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('FixedGridImportanceSampler')
        prefix = '--importance-sampler:grid:'
        group.add_argument(prefix + 'grid-size', type=str, default=None, help="""
        grid resolution for loss-importance-based resampling of the dataset
        Example: 64:64:64
        """)
        group.add_argument(prefix + 'num-samples-per-voxel', type=int, default=8, help="""
        number of samples per voxel for loss evaluation in importance-based dataset resampling
        """)
        group.add_argument(prefix + 'min-density', type=float, default=0.01, help="""
        minimum density for sampling per voxel 
        """)
        group.add_argument(prefix + 'batch-size', type=int, default=None, help="""
        batch size during importance sampling
        """)
        group.add_argument(prefix + 'sub-sampling', type=int, default=None, help="""
        sub-sampling factor to use for grid resolutions
        """)

    @classmethod
    def from_dict(cls, args: Dict[str, Any], dimension=None, sampler: Optional[ISampler] = None, device=None):
        sampler_kws = cls.read_args(args)
        return cls(sampler=sampler, dimension=dimension, **sampler_kws, device=device)

    @staticmethod
    def read_args(args):
        prefix = 'importance_sampler:grid:'
        out = {
            key: args[prefix + key]
            for key in ['grid_size', 'num_samples_per_voxel', 'min_density', 'batch_size', 'sub_sampling']
        }
        if out['grid_size'] is not None:
            gs = out['grid_size'].split(':')
            assert len(gs) in {1, 3}
            if len(gs) == 1:
                gs = [gs[0]] * 3
            gs = (int(gs[0]), int(gs[1]), int(gs[2]))
            out['grid_size'] = gs
        return out

    def __init__(
            self,
            grid_size: Optional[Tuple[int, ...]] = None, sub_sampling: Optional[int] = None, sampler: Optional[ISampler] = None,
            num_samples_per_voxel=8, min_density=0.01,
            batch_size=None, root_box: Optional[CoordinateBox] = None, dimension: Optional[int] = None,
            verbose=False, device=None,
    ):
        assert (grid_size is not None) or (sub_sampling is not None), '[ERROR] Grid size or data-grid subsampling parameter must be given to FixedGridImportanceSampler'
        assert (grid_size is None) or (sub_sampling is None)
        if dimension is None and grid_size is not None:
            dimension = len(grid_size)
        if dimension is None and sampler is not None:
            dimension = sampler.dimension
        if dimension is None and root_box is not None:
            dimension = root_box.dimension
        assert dimension is not None
        super(FixedGridImportanceSampler, self).__init__(dimension, root_box, device)
        if grid_size is not None:
            assert dimension == len(grid_size)
        if sampler is not None:
            assert dimension == sampler.dimension
        else:
            assert device is not None
            sampler = RandomUniformSampler(dimension, self.device)
        if root_box is not None:
            assert dimension == root_box.dimension
        assert sampler.device == self.device

        self.sampler = sampler
        self.grid_size = grid_size
        self.sub_sampling = sub_sampling
        self.root_box = root_box
        self.num_samples_per_voxel = num_samples_per_voxel
        self.min_density = min_density
        self.batch_size = batch_size
        self.verbose = True

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator, grid_size: Optional[Tuple[int, ...]] = None, **kwargs):
        with torch.no_grad():
            if self.grid_size is None:
                assert grid_size is not None
                grid_size = tuple(int(np.ceil(s / self.sub_sampling)) for s in grid_size)
            else:
                grid_size = self.grid_size
            assert grid_size is not None and len(grid_size) == self.dimension
            stratified_sampler = StratifiedGridSampler(self.sampler, grid_size, handle_remainder=False)
            value_grid = self._build_value_grid(evaluator, stratified_sampler)
            importance_grid = torch.maximum(value_grid / torch.max(value_grid).item(), torch.tensor([self.min_density], device=value_grid.device))
            positions = self._generate_importance_samples(num_samples, importance_grid)
            if self.root_box is not None:
                positions = self.root_box.rescale(positions)
        return positions

    def _generate_importance_samples(self, num_samples: int, importance_grid: Tensor):
        sampler = RandomUniformSampler(self.dimension, self.sampler.device)
        batch_size = self.batch_size if self.batch_size is not None else num_samples
        all_samples = []
        all_weights = []
        total_samples = 0
        total_accepted_samples = 0
        current_sample_count = 0
        while current_sample_count < num_samples:
            samples = sampler.generate_samples(batch_size)
            threshold = F.grid_sample(
                importance_grid[None, None, ...],
                2. * torch.flip(samples, [-1]).view(*[1 for _ in importance_grid.shape], *samples.shape) - 1.,
                align_corners=True, mode='bilinear'
            )
            threshold = threshold.view(-1)
            accepted = (torch.rand(batch_size, device=sampler.device) < threshold)
            samples = samples[accepted]
            num_accepted = len(samples)
            total_samples = total_samples + batch_size
            if num_accepted > 0:
                total_accepted_samples = total_accepted_samples + num_accepted
                if current_sample_count + num_accepted > num_samples:
                    samples = samples[:(num_samples - current_sample_count)]
                all_samples.append(samples)
                all_weights.append(1. / threshold[accepted])
                current_sample_count = current_sample_count + len(samples)
        if self.verbose:
            self._print_statistics(total_samples, total_accepted_samples)
        samples = torch.cat(all_samples, dim=0)
        weights = torch.cat(all_weights,dim=0)
        return samples, weights

    def _print_statistics(self, total_samples, total_accepted_samples):
        print(
            '[INFO] Finished importance sampling after {tot} total samples. Acceptance rate was {frac:.4f}'.format(
                tot=total_samples, frac=total_accepted_samples / total_samples
            )
        )

    def _build_value_grid(self, evaluator: IFieldEvaluator, stratified_sampler: StratifiedGridSampler):
        assert evaluator.device == stratified_sampler.device
        grid_size = stratified_sampler.grid_size
        grid_numel = int(np.prod(grid_size))
        grid = torch.zeros(*grid_size, dtype=torch.float32, device=evaluator.device)
        for j in range(self.num_samples_per_voxel):
            positions = stratified_sampler.generate_samples(grid_numel)
            values = evaluator.evaluate(positions)
            grid += values.view(*grid_size)
        return grid


def _test_importance_sampler():
    import matplotlib.pyplot as plt

    class Evaluator(IFieldEvaluator):

        def __init__(self, dimension, device=None):
            super(Evaluator, self).__init__(dimension, 1, device)
            self.direction = 4 * torch.tensor([1] * dimension, device=device)[None, :]# torch.randn(1, dimension, device=device)
            self.offset = torch.tensor([0.5] * dimension, device=device)[None, :] # torch.randn(1, dimension, device=device)

        def forward(self, positions: Tensor) -> Tensor:
            return torch.exp(2 * torch.sum(self.direction * (positions - self.offset), dim=-1))

    dimension = 2
    device = torch.device('cuda:0')
    eval = Evaluator(dimension, device)
    sampler = FixedGridImportanceSampler((16, 16), verbose=True, device=device, min_density=0.001)
    positions = sampler.generate_samples(100000, eval)
    values = eval.evaluate(positions)[:, 0]
    positions = positions.data.cpu().numpy()
    values = values.data.cpu().numpy()
    plt.scatter(positions[:, 0], positions[:, 1], c=values, alpha=0.05)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.close()

    print('Finished')


def _test_sampler():
    import torch
    from torch import Tensor

    class Evaluator(IFieldEvaluator):

        def __init__(self, dimension, device=None):
            super(Evaluator, self).__init__(dimension, 1, device)
            self.direction = 4 * torch.tensor([1] * dimension, device=device)[None, :]# torch.randn(1, dimension, device=device)
            self.offset = torch.tensor([0.5] * dimension, device=device)[None, :] # torch.randn(1, dimension, device=device)

        def forward(self, positions: Tensor) -> Tensor:
            return torch.sum(self.direction * (positions - self.offset), dim=-1) ** 2

    device = torch.device('cuda:0')
    evaluator = Evaluator(3, device=device)
    sampler = FixedGridImportanceSampler(
        dimension=3, device=device,
        min_density=0.01, grid_size=(64, 64, 64)
    )

    for i in range(20):
        samples = sampler.generate_samples(4000, evaluator)
        c = evaluator.evaluate(samples)
        samples = samples.data.cpu().numpy()
        c = c[:, 0].data.cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=c)
        plt.show()
        plt.close()


if __name__ == '__main__':
    _test_sampler()
