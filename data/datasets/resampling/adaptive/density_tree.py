from typing import Optional, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import patches

from inference.field_evaluator import IFieldEvaluator
from data.datasets.resampling.coordinate_box import CoordinateBox, UnitCube
from data.datasets.resampling.adaptive.data_summaries import SampleSummary, MergerSummary
from data.datasets.resampling.adaptive.statistical_tests import (
    FastWhiteHomoscedasticityTest,
    FastKolmogorovSmirnovTestNd
)

from data.datasets.sampling.random_uniform import RandomUniformSampler


class NodeSet(object):

    @classmethod
    def merge(cls, *levels: 'NodeSet'):
        dimension = levels[0].dimension
        dimensions, lowers, extents, parents, data, queried, splitted, devices = list(zip(*[
            (l.dimension, l.lower, l.extent, l.parents, l.sample_summary, l.children_summary, l.split_dimension, l.device)
            for l in levels
        ]))
        assert np.all(np.array(dimensions) == dimension)
        parents_is_none = np.array([p is None for p in parents])
        has_parents = np.all(~parents_is_none)
        data_is_none = np.array([d is None for d in data])
        has_data = np.all(~data_is_none)
        queried_is_none = np.array([q is None for q in queried])
        has_queried = np.all(~queried_is_none)
        if not (has_parents or np.all(parents_is_none)):
            raise Exception()
        assert has_data or np.all(data_is_none)
        assert has_queried or np.all(queried_is_none)
        device = devices[0]
        assert np.all([d == device for d in devices])
        # implementation does currently not check overlap between nodes!
        out = cls(dimension, torch.cat(lowers, dim=0), torch.cat(extents, dim=0), parents=torch.cat(parents, dim=0), split_dimensions=torch.cat(splitted, dim=0), device=device)
        if has_data:
            out.sample_summary = SampleSummary.merge(*data)
        if has_queried:
            out.children_summary = MergerSummary.merge(*[(q, None) for q in queried])
        return out

    @staticmethod
    def _to_device(x: torch.Tensor, device):
        if device is None or x.device == device:
            return x
        return x.to(device)

    def __init__(
            self,
            dimension: int, lower: torch.Tensor, extent: torch.Tensor,
            parents: Optional[torch.Tensor] = None, superset_index: Optional[torch.Tensor] = None,
            split_dimensions: Optional[torch.Tensor] = None,
            device=None
    ):
        assert len(lower) == len(extent), \
            f'[ERROR] Obtained inconsistent specifications for node lower bounds (length {len(lower)}) and extents (length {len(extent)}).'
        self.dimension = dimension
        assert lower.shape[-1] == self.dimension and len(lower.shape) == 2, \
            f'[ERROR] Shape of lower bounds {lower.shape} is not consistent with dimension {dimension}.'
        self.lower = self._to_device(lower, device)
        assert extent.shape[-1] == self.dimension and len(extent.shape) == 2, \
            f'[ERROR] Shape of extents {extent.shape} is not consistent with dimension {dimension}.'
        self.extent = self._to_device(extent, device)
        if parents is not None:
            assert len(parents) == lower.shape[0]
            parents = self._to_device(parents, device)
        self.parents = parents
        if superset_index is not None:
            assert len(superset_index) == lower.shape[0]
            superset_index = self._to_device(superset_index, device)
        self.superset_index = superset_index
        if split_dimensions is not None:
            assert len(split_dimensions) == lower.shape[0]
        else:
            split_dimensions = - torch.ones(len(extent), dtype=torch.long, device=lower.device)
        self.split_dimension = self._to_device(split_dimensions, device)
        self.device = device
        self.sample_summary = None
        self.children_summary = None

    def num_nodes(self):
        return len(self.lower)

    def lower_bounds(self, index: Optional[torch.Tensor]=None):
        return self._select_if_required(self.lower, index)

    def upper_bounds(self, index: Optional[torch.Tensor]=None):
        return self._select_if_required(self.lower, index) + self._select_if_required(self.extent, index)

    def centers(self, index: Optional[torch.Tensor]=None):
        return self._select_if_required(self.lower, index) + self._select_if_required(self.extent, index) / 2.

    def volume(self, index: Optional[torch.Tensor] = None):
        return torch.prod(self._select_if_required(self.extent, index), dim=-1)

    @staticmethod
    def _select_if_required(x: torch.Tensor, index: Union[torch.Tensor, None]):
        return x if index is None else x[index]

    def get_subset(self, index: torch.Tensor):
        if self.parents is not None:
            parents = self.parents[index]
        else:
            parents = None
        if self.superset_index is not None:
            superset_index = self.superset_index[index]
        else:
            superset_index = torch.arange(self.num_nodes())[index]
        out = NodeSet(
            self.dimension, self.lower[index], self.extent[index],
            parents=parents, superset_index=superset_index,
            split_dimensions=self.split_dimension[index],
            device=self.device
        )
        if self.sample_summary is not None:
            leaf_nodes = -torch.ones_like(self.split_dimension)
            num_leaf_nodes = torch.sum(self.split_dimension < 0).item()
            leaf_nodes[self.split_dimension < 0] = torch.arange(num_leaf_nodes, device=self.device)
            leaf_nodes = leaf_nodes[index][self.split_dimension[index] < 0]
            out.sample_summary = self.sample_summary.get_subset(leaf_nodes)
        if self.children_summary is not None:
            out.children_summary = self.children_summary.get_subset(index)
        return out

    def rescale(self, x: torch.Tensor, index=None):
        lower = self._select_if_required(self.lower, index)
        extent = self._select_if_required(self.extent, index)
        self_shape = lower.shape
        x_shape = x.shape
        assert x_shape[-2:] == self_shape, \
            f'[ERROR] Input shape {x.shape} is inconsistent with expected shape {self_shape}.'
        if len(x_shape) > 2:
            new_axes = tuple(1 for _ in x_shape[:-len(self_shape)])
            lower = lower.view(new_axes + self_shape)
            extent = extent.view(new_axes + self_shape)
        return extent * x + lower

    def store_sample_summary(self, samples: torch.Tensor):
        assert samples.shape[-1] == torch.sum(self.split_dimension < 0).item()
        new_data = SampleSummary.from_sample(samples)
        if self.sample_summary is None:
            self.sample_summary = new_data
        else:
            self.sample_summary = SampleSummary.from_sample_summaries(self.sample_summary, new_data)
        return self

    def split_along_dimensions(self, dimensions: torch.Tensor, index: Optional[torch.Tensor] = None):
        new_extent = self._select_if_required(self.extent, index).clone()
        assert len(new_extent) == len(dimensions)
        idx = torch.arange(len(new_extent))
        offset = new_extent[idx, dimensions] / 2.
        new_extent[idx, dimensions] = offset
        lower = self._select_if_required(self.lower, index).clone()
        upper = lower.clone()
        upper[idx, dimensions] = lower[idx, dimensions] + offset
        lower = torch.reshape(torch.stack([lower, upper], dim=1), (-1, self.dimension))
        new_extent = torch.repeat_interleave(new_extent, 2, dim=0)
        if self.superset_index is not None:
            parents = self.superset_index[index]
        else:
            parents = index
        parents = torch.repeat_interleave(parents, 2, dim=0)
        self.split_dimension[index] = dimensions
        return NodeSet(self.dimension, lower, new_extent, parents=parents, device=self.device)

    def add_children_summary(self, summary: Union[MergerSummary, None]):
        assert self.children_summary is None
        is_leaf_node = self.split_dimension < 0
        if torch.all(is_leaf_node):
            if self.sample_summary is None:
                raise Exception()
            self.children_summary = MergerSummary.merge((self.sample_summary, self.volume()))
            return self
        assert summary is not None
        if torch.any(is_leaf_node):
            assert self.sample_summary is not None
            self.children_summary = MergerSummary.merge(
                (summary, None),
                (self.sample_summary, self.volume(index=is_leaf_node)),
                index=is_leaf_node.to(dtype=torch.long)
            )
        else:
            self.children_summary = summary
        return self

    def summarize_nodes(self):
        assert  self.children_summary is not None
        index = torch.arange(self.num_nodes(), device=self.device)
        i1, i2 = index[::2], index[1::2]
        v = self.volume()
        v1 = v[i1]
        v2 = v[i2]
        s1 = self.children_summary.get_subset(i1)
        s2 = self.children_summary.get_subset(i2)
        return MergerSummary.from_summaries(s1, v1, s2, v2)

    def plot(self, ax=None):
        internal_ax = ax is None
        if internal_ax:
            _, ax = plt.subplots(1, 1)
        for xy, width in zip(self.lower.data.cpu().numpy(), self.extent.data.cpu().numpy()):
            rectangle = patches.Rectangle(xy, *width.tolist(), edgecolor='black')
            ax.add_patch(rectangle)
        if internal_ax:
            plt.show()
            plt.close()


class FastDensityTree(object):

    @classmethod
    def from_scalar_field(
            cls, root_box: CoordinateBox,
            sampler, field_evaluator: IFieldEvaluator, difference_test,
            homoscedasticity_test=None,
            min_depth=0, max_depth=16,
            num_samples_per_node=64, store_sample_summary=True,
            num_samples_per_batch=64 ** 3,
            device=None
    ):
        assert root_box.device == sampler.device, f'[ERROR] Root box and sampler must be located on the same device. Found root box on {root_box.device} and sampler on {sampler.device}.'
        device = root_box.device
        current_depth = 0
        root_set = NodeSet(
            root_box.dimension,
            root_box.lower_bounds(keepdim=True),
            root_box.size(keepdim=True),
            device=device
        )
        levels = {current_depth: root_set}
        while current_depth in levels:
            current_set = levels[current_depth]
            num_nodes = current_set.num_nodes()
            num_nodes_per_batch = num_samples_per_batch // num_samples_per_node
            node_index = torch.arange(num_nodes, device=device)
            if num_nodes > num_nodes_per_batch:
                node_index = torch.chunk(node_index, int(np.ceil(num_nodes / num_nodes_per_batch)))
                batches = [current_set.get_subset(i) for i in node_index]
            else:
                batches = [current_set]
            children = []
            for batch in batches:
                num_current_nodes = batch.num_nodes()
                raw_samples = sampler.generate_samples(num_current_nodes * num_samples_per_node)
                if type(raw_samples).__module__ == 'numpy':
                    raw_samples = torch.from_numpy(raw_samples)
                if raw_samples.device != device:
                    raw_samples = raw_samples.to(device)
                raw_samples = raw_samples.view(num_samples_per_node, num_current_nodes, batch.dimension)
                positions = batch.rescale(raw_samples).view(-1, batch.dimension)
                values = field_evaluator.evaluate(positions).view(num_samples_per_node, num_current_nodes)
                positions = positions.view(num_samples_per_node, num_current_nodes, batch.dimension)
                split_dimension = - torch.ones(num_current_nodes, dtype=torch.long, device=device)
                if current_depth < max_depth:
                    classification = positions < batch.centers()[None, ...]
                    difference_result = difference_test.compute(values, classification)
                    best_split = difference_result.best_split()
                    difference_reject = difference_result.reject()
                    if min_depth is not None and current_depth <= min_depth:
                        split_dimension = difference_result.best_split()
                    elif torch.any(difference_reject):
                        split_dimension[difference_reject] = best_split[difference_reject]
                    difference_no_reject = ~ difference_reject
                    if torch.any(difference_no_reject) and homoscedasticity_test is not None:
                        homoscedasticity_result = homoscedasticity_test.compute(
                            values[:, difference_no_reject], positions[:, difference_no_reject]
                        )
                        homoscedasticity_reject = homoscedasticity_result.reject()
                        if torch.any(homoscedasticity_reject):
                            secondary_reject = torch.full_like(difference_no_reject, False)
                            secondary_reject[difference_no_reject] = homoscedasticity_reject
                            split_dimension[secondary_reject] = best_split[secondary_reject]
                split_required = split_dimension >= 0
                if torch.any(split_required):
                    split_index = torch.arange(num_current_nodes, device=device)[split_required]
                    batch_children = batch.split_along_dimensions(split_dimension[split_required], index=split_index)
                    children.append(batch_children)
                is_leaf_node = ~split_required
                if store_sample_summary and torch.any(is_leaf_node):
                    batch.store_sample_summary(values[:, is_leaf_node])
            if store_sample_summary and len(batches) > 1:
                levels[current_depth] = NodeSet.merge(*batches) # to save data
            if len(children) > 1:
                levels[current_depth + 1] = NodeSet.merge(*children)
            elif len(children) == 1:
                levels[current_depth + 1] = children[0]
            else:
                pass
            current_depth = current_depth + 1
        return cls(levels)

    def __init__(self, levels: Dict[int, NodeSet]):
        assert 0 in levels
        self.dimension = levels[0].dimension
        for l in levels.values():
            assert l.dimension == self.dimension
        self.levels = levels

    def add_summaries(self):
        summary = None
        for key in reversed(sorted(self.levels.keys())):
            self.levels[key].add_children_summary(summary)
            if key - 1 in self.levels:
                summary = self.levels[key].summarize_nodes()
        return self

    def depth(self):
        return len(self.levels) - 1

    def plot(self):
        for key in sorted(self.levels.keys()):
            self.levels[key].plot()

    def get_leaf_nodes(self):
        leafs = []
        for key in sorted(self.levels.keys()):
            level = self.levels[key]
            is_leaf_node = level.split_dimension < 0
            if torch.any(is_leaf_node):
                leafs.append(level.get_subset(is_leaf_node))
        return NodeSet.merge(*leafs)

# def _test_ks_test():
#     from data.datasets.resampling.adaptive.legacy.statistical_tests import MyKolmogorovSmirnovTestNd
#     test_old = MyKolmogorovSmirnovTestNd(alpha=0.05)
#     test_new = FastKolmogorovSmirnovTestNd(alpha=0.05)
#     samples = np.exp(np.random.randn(10, 2))
#     classification = np.concatenate([np.zeros((5, 2, 3)), np.ones((5, 2, 3))], axis=0).astype(bool)
#     r1 = test_old.compute(samples, classification)
#     print(r1.test_statistics, r1.significance_ratios)
#     with torch.no_grad():
#         r2 = test_new.compute(torch.from_numpy(samples), torch.from_numpy(classification))
#         print(r2.test_statistics, r2.significance_ratios)
#
#
# def _test_white_test():
#     from data.datasets.resampling.adaptive.legacy.statistical_tests import MyWhiteHomoscedasticityTest
#     test_old = MyWhiteHomoscedasticityTest(alpha=0.05, simplify_predictors=True)
#     test_new = FastWhiteHomoscedasticityTest(alpha=0.05, simplify_predictors=True)
#     samples = np.exp(np.random.randn(10, 2))
#     coordinates = np.random.rand(10, 2, 3)
#     r1 = test_old.compute(samples, coordinates)
#     print(r1.test_tatistic, r1.p_value)
#     with torch.no_grad():
#         r2 = test_new.compute(torch.from_numpy(samples), torch.from_numpy(coordinates))
#         print(r2.test_tatistic, r2.p_value)


def _test_tree():
    class Evaluator(object):

        def __init__(self, dimension, noise_amplitude=0.1, seed=None):
            self.dimension = dimension
            self.rng = np.random.Generator(np.random.PCG64(seed))
            self.v_mu = self.rng.normal(size=(1, dimension))
            self.v_beta = self.rng.random(size=(1, dimension))
            self.noise_amplitude = noise_amplitude

        def evaluate(self, coordinates: torch.Tensor):
            mu = np.sum(self.v_mu * (coordinates.data.cpu().numpy() - self.v_beta), axis=-1)
            # log_std = np.sum(self.v_log_std * coordinates, axis=-1)
            z = mu  # + np.random.randn(*mu.shape) * np.exp(log_std) / 20.
            return torch.from_numpy(z ** 2 + self.noise_amplitude * self.rng.normal(size=z.shape)).to(device=coordinates.device)

    dimension = 2
    seed= 42
    alpha=0.05
    torch.random.manual_seed(seed)
    box = UnitCube(dimension)
    with torch.no_grad():
        tree = FastDensityTree.from_scalar_field(
            box, RandomUniformSampler(dimension), Evaluator(dimension, seed=seed, noise_amplitude=0.01),
            FastKolmogorovSmirnovTestNd(alpha=alpha), FastWhiteHomoscedasticityTest(alpha=alpha),
            min_depth=2, max_depth=10,
            num_samples_per_node=2048, num_samples_per_batch=64**3
        )
        tree.add_summaries()
    print('Finished')


if __name__ == '__main__':
    _test_tree()