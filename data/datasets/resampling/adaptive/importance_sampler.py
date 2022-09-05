import argparse
from typing import Optional

import torch
from matplotlib import pyplot as plt

from common.mathparser import BigInteger
from inference import IFieldEvaluator
from data.datasets.resampling.coordinate_box import CoordinateBox, UnitCube
from data.datasets.resampling.adaptive.density_tree_sampler import FastDensityTreeSampler
from data.datasets.resampling.adaptive.density_tree import FastDensityTree
from data.datasets.resampling.adaptive.statistical_tests import (
    FastKolmogorovSmirnovTestNd,
    FastWhiteHomoscedasticityTest,
)
from data.datasets.resampling import IImportanceSampler
from data.datasets.sampling import ISampler, RandomUniformSampler


class DensityTreeImportanceSampler(IImportanceSampler):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('DensityTreeImportanceSampler')
        prefix = '--importance-sampler:tree:'
        group.add_argument(prefix + 'min-depth', type=int, default=4, help="""
        minimum tree depth for adaptive loss grid
        """)
        group.add_argument(prefix + 'max-depth', type=int, default=12, help="""
        maximum tree depth for adaptive loss grid
        """)
        group.add_argument(prefix + 'num-samples-per-node', type=int, default=128, help="""
        number of samples per node for loss tree refinement
        """)
        group.add_argument(prefix + 'alpha', type=float, default=0.05, help="""
        significance threshold for splitting decision
        """)
        group.add_argument(prefix + 'batch-size', type=BigInteger, default=None, help="""
        batch size for loss evaluation during importance sampling (Default: Dataset batch size)
        """)
        group.add_argument(prefix + 'min-density', type=float, default=0.01, help="""
        minimum probability density for sampling per grid box 
        """)
        group.add_argument(prefix + 'max-ratio', type=float, default=10, help="""
        maximum ratio of probability densities during node splitting 
        """)
        # group.add_argument(prefix + 'seed', type=int, default=42, help="""
        # seed for importance sampling random number generator
        # """)

    @staticmethod
    def read_args(args: dict):
        prefix = 'importance_sampler:tree:'
        return {
            key: args[prefix + key]
            for key in ['min_depth', 'max_depth', 'num_samples_per_node', 'batch_size',
                        'min_density', 'max_ratio', 'alpha']
        }

    @classmethod
    def from_dict(cls, args, dimension=None, device=None):
        sampler_kws = DensityTreeImportanceSampler.read_args(args)
        return DensityTreeImportanceSampler(**sampler_kws, dimension=dimension, device=device)

    def __init__(
            self,
            sampler: Optional[ISampler] = None, dimension:Optional[int] = None, batch_size=None,
            min_depth=4, max_depth=8, num_samples_per_node=128, min_density=0.01, max_ratio=10,
            alpha=0.05, root_box: Optional[CoordinateBox] = None, device=None, dtype=None, seed=None
    ):
        if dimension is None and sampler is not None:
            dimension = sampler.dimension
        if dimension is None and root_box is not None:
            dimension = root_box.dimension
        assert dimension is not None
        if sampler is not None:
            assert dimension == sampler.dimension
            if device is not None:
                assert device == sampler.device
            if dtype is not None:
                assert dtype == sampler.dtype
            device = sampler.device
            dtype = sampler.dtype
        else:
            assert device is not None
            sampler = RandomUniformSampler(dimension, device=device, dtype=dtype)
        if root_box is not None:
            assert dimension == root_box.dimension
            assert device == root_box.device
        else:
            root_box = UnitCube(dimension, device=device, dtype=dtype)
        super(DensityTreeImportanceSampler, self).__init__(dimension, root_box, device)
        self.dimension = dimension
        self.sampler = sampler
        self.root_box = root_box
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_samples_per_node = num_samples_per_node
        self.min_density = min_density
        self.max_ratio = max_ratio
        self.alpha = alpha
        assert batch_size is not None
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = device

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator, **kwargs):
        with torch.no_grad():
            difference_test = FastKolmogorovSmirnovTestNd(alpha=self.alpha)
            homoscedasticity_test = FastWhiteHomoscedasticityTest(alpha=self.alpha)
            tree = FastDensityTree.from_scalar_field(
                self.root_box, self.sampler, evaluator, difference_test, homoscedasticity_test=homoscedasticity_test,
                min_depth=self.min_depth,max_depth=self.max_depth, num_samples_per_node=self.num_samples_per_node,
                store_sample_summary=True, num_samples_per_batch=self.batch_size,
                device=self.device
            )
            tree.add_summaries()
            sampler = FastDensityTreeSampler(
                self.sampler, tree,
                min_density=self.min_density, max_ratio=self.max_ratio
            )
            samples, weights = sampler.generate_samples(num_samples)
            if samples.device != self.device:
                samples = samples.to(self.device)
                weights = weights.to(self.device)
            perm = torch.randperm(num_samples, device=torch.device('cpu'))
            samples = samples[perm]
            weights = weights[perm]
        return samples, weights


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
    sampler = DensityTreeImportanceSampler(
        dimension=3, device=device, batch_size=64**3,
        max_ratio=3,
        min_density=0.1
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