from typing import Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from data.datasets.resampling.adaptive.density_tree import FastDensityTree, NodeSet
from data.datasets.sampling import ISampler, RandomUniformSampler


class FastDensityTreeSampler(ISampler):

    def __init__(self, sampler: ISampler, tree: FastDensityTree, min_density: Optional[float] = None, max_ratio: Optional[float] = None):
        super(FastDensityTreeSampler, self).__init__(tree.dimension, sampler.device, sampler.dtype)
        self.sampler = sampler
        self.tree = tree
        self.min_density = min_density
        self.max_ratio = np.exp(np.abs(np.log(max_ratio)))

    def generate_samples(self, num_samples: int):

        densities = 1. / self.tree.levels[0].volume()
        active_nodes = torch.full(densities.shape, True, device=densities.device)
        counts = torch.full(densities.shape, num_samples, device=densities.device)

        current_depth = 0
        samples = []
        weights = []
        while torch.any(active_nodes):
            current_level = self.tree.levels[current_depth]
            active_subset = current_level.get_subset(active_nodes)
            needs_samples = active_subset.split_dimension < 0
            if self.min_density is not None:
                needs_samples = torch.logical_or(needs_samples, densities <= self.min_density)
            needs_propagation = ~ needs_samples
            if torch.any(needs_samples):
                sample_counts = counts[needs_samples]
                sample_subset = active_subset.get_subset(needs_samples)
                samples.append(self._draw_samples_from_subset(sample_subset, sample_counts))
                weights.append(self._compute_weights(densities[needs_samples], sample_counts))
                active_nodes[active_nodes.clone()] = needs_propagation
                counts = counts[needs_propagation]
                densities = densities[needs_propagation]
            if torch.any(needs_propagation):
                active_nodes = torch.repeat_interleave(active_nodes[current_level.split_dimension >= 0], 2)
                child_level = self.tree.levels[current_depth + 1]
                active_subset = child_level.get_subset(active_nodes)
                node_index = torch.arange(active_subset.num_nodes())
                c1 = active_subset.get_subset(node_index[::2])
                c2 = active_subset.get_subset(node_index[1::2])
                data = torch.stack([c1.children_summary.mean(), c2.children_summary.mean()], dim=0)
                p = self._apply_constraints(data / torch.sum(data, dim=0, keepdim=True), densities)
                densities = 2. * densities[None, ...] * p
                densities = densities.T.reshape(-1)
                dist = torch.distributions.Binomial(counts, probs=p[0])
                counts1 = dist.sample()
                counts2 = counts - counts1
                counts = torch.stack([counts1, counts2], dim=-1).view(-1)
                is_nonzero = counts > 0
                active_nodes[active_nodes.clone()] = is_nonzero
                counts = counts[is_nonzero]
                densities = densities[is_nonzero]
            current_depth = current_depth + 1
        samples = torch.cat(samples, dim=0)
        weights = torch.cat(weights, dim=0)
        return samples, weights

    def _draw_samples_from_subset(self, node_set: NodeSet, counts: Tensor):
        assert len(counts) == node_set.num_nodes()
        counts = counts.to(dtype=torch.long)
        total_samples = torch.sum(counts).item()
        assert int(total_samples) == total_samples
        raw_samples = self.sampler.generate_samples(int(total_samples))
        raw_samples = torch.split(raw_samples, counts.tolist(), dim=0)
        extent = node_set.extent
        lower = node_set.lower
        samples = [
            e[None, ...] * s + l[None, ...]
            for e, l, s in zip(extent, lower, raw_samples)
        ]
        samples = torch.cat(samples, dim=0)
        return samples

    def _compute_weights(self, densities, counts):
        device = densities.device
        dtype = densities.dtype
        all_densities = [
            torch.full((int(c.item()),), d.item(), dtype=dtype, device=device)
            for d, c in zip(densities, counts)
        ]
        return 1. / torch.cat(all_densities).view(-1)

    def _apply_constraints(self, p_raw: Tensor, densities: Tensor):
        constraints = torch.zeros_like(p_raw[0])
        def update_constraints(new_constraint, constraints):
            if not isinstance(new_constraint, Tensor):
                new_constraint = torch.ones_like(constraints) * new_constraint
            return torch.maximum(new_constraint, constraints)

        if self.min_density is not None:
            constraints = update_constraints(self.min_density / (2. * densities), constraints)
        if self.max_ratio is not None:
            new_constraint = torch.tensor([1. / (1 + self.max_ratio)], dtype=constraints.dtype, device=constraints.device)
            constraints = update_constraints(new_constraint, constraints)
        p = p_raw
        if constraints is not None:
            idx_lower = torch.argmin(p_raw, dim=0)
            idx_higher = 1 - idx_lower
            idx_nodes = torch.arange(p_raw.shape[-1])
            needs_update = p_raw[idx_lower, idx_nodes] < constraints
            p[idx_lower, idx_nodes] = torch.where(needs_update, constraints, p_raw[idx_lower, idx_nodes])
            p[idx_higher, idx_nodes] = torch.where(needs_update, 1. - constraints, p_raw[idx_higher, idx_nodes])
        return p


def _test_sampler():

    from data.datasets.resampling.coordinate_box import UnitCube
    from data.datasets.resampling.adaptive.statistical_tests import FastKolmogorovSmirnovTestNd
    from data.datasets.resampling.adaptive.statistical_tests import FastWhiteHomoscedasticityTest
    from inference import IFieldEvaluator

    class Evaluator(IFieldEvaluator):

        def __init__(self, dimension, noise_amplitude=0.1, seed=None):
            super(Evaluator, self).__init__(dimension, 1, None)
            self.dimension = dimension
            self.rng = np.random.Generator(np.random.PCG64(seed))
            self.v_mu = self.rng.normal(size=(1, dimension))
            self.v_beta = self.rng.random(size=(1, dimension))
            self.noise_amplitude = noise_amplitude

        def evaluate(self, coordinates: torch.Tensor):
            mu = np.sum(self.v_mu * (coordinates.data.cpu().numpy() - self.v_beta), axis=-1)
            # log_std = np.sum(self.v_log_std * coordinates, axis=-1)
            z = mu  # + np.random.randn(*mu.shape) * np.exp(log_std) / 20.
            return torch.from_numpy(z ** 2 + self.noise_amplitude * self.rng.normal(size=z.shape)).to(device=coordinates.device, dtype=coordinates.dtype)

    dimension = 2
    seed= 42
    alpha=0.01
    num_samples = 10000
    device = torch.device('cuda:0')
    dtype = torch.float32
    evaluator = Evaluator(dimension, seed=seed, noise_amplitude=0.01)

    box = UnitCube(dimension, device=device, dtype=dtype)
    with torch.no_grad():
        tree = FastDensityTree.from_scalar_field(
            box, RandomUniformSampler(dimension, device=device, dtype=dtype), evaluator,
            FastKolmogorovSmirnovTestNd(alpha=alpha),
            FastWhiteHomoscedasticityTest(alpha=alpha),
            min_depth=2, max_depth=16,
            num_samples_per_node=128, num_samples_per_batch=64**3,
        )
        tree.add_summaries()
        leafs = tree.get_leaf_nodes()
        leafs.plot()
        sampler = FastDensityTreeSampler(
            RandomUniformSampler(dimension, device=device, dtype=dtype), tree,
            min_density=0.01, max_ratio=10
        )
        samples = sampler.generate_samples(num_samples)
        values = evaluator.evaluate(samples)
    samples = samples.data.cpu().numpy()
    values = values.data.cpu().numpy()
    plt.scatter(samples[:,0], samples[:, 1], c=values, alpha=0.1)
    plt.show()
    print('Finished')


if __name__ == '__main__':
    _test_sampler()
