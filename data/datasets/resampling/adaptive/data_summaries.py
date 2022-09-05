from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor


class _SummaryStatistics(object):

    def mean(self):
        raise NotImplementedError()

    def var(self, unbiased=True):
        raise NotImplementedError()

    def std(self, unbiased=True):
        return torch.sqrt(self.var(unbiased=unbiased))

    def num_samples(self):
        raise NotImplementedError()

    def var_of_mean(self):
        raise NotImplementedError()

    def std_of_mean(self):
        return torch.sqrt(self.var_of_mean())

    def min(self):
        raise NotImplementedError()

    def max(self):
        raise NotImplementedError()


class SampleSummary(_SummaryStatistics):

    def __init__(self, num_samples: Tensor, mu: Tensor, sum_of_squares: Tensor, min: Tensor, max: Tensor):
        length = len(num_samples)
        assert np.all([len(x) == length for x in [mu, sum_of_squares, min, max]]), \
            '[ERROR] Data statistics must have the same number of entries.'
        self._num_samples = num_samples
        self._mu = mu
        self._sum_of_squares = sum_of_squares
        self._min = min
        self._max = max

    @classmethod
    def from_sample(cls, sample: torch.Tensor):
        mu = torch.mean(sample, dim=0)
        sum_of_squares = torch.sum(torch.square(torch.abs(sample - mu)), dim=0)
        return cls(
            torch.ones(len(mu), dtype=torch.long, device=mu.device) * len(sample),
            mu, sum_of_squares, torch.amin(sample, dim=0), torch.amax(sample, dim=0)
        )

    @classmethod
    def from_sample_summaries(cls, stats1: 'SampleSummary', stats2: 'SampleSummary'):
        num_samples = stats1._num_samples + stats2._num_samples
        f = stats1._num_samples * stats2._num_samples / num_samples
        mu1, mu2 = stats1.mean(), stats2.mean()
        delta = torch.abs(mu2 - mu1)
        mu = (stats1._num_samples * mu1 + stats2._num_samples * mu2) / num_samples
        sum_of_squares = stats1._sum_of_squares + stats2._sum_of_squares + f * delta ** 2
        min_ = torch.minimum(stats1.min(), stats2.max())
        max_ = torch.maximum(stats1.max(), stats2.max())
        return cls(num_samples, mu, sum_of_squares, min_, max_)

    @classmethod
    def merge(cls, *summaries: 'SampleSummary'):
        fields = list(zip(*[
            (s._num_samples, s._mu, s._sum_of_squares, s._min, s._max)
            for s in summaries
        ]))
        return cls(*[torch.cat(f, dim=0) for f in fields])

    def get_subset(self, index: Tensor):
        return SampleSummary(*[
            x[index]
            for x in [self._num_samples, self._mu, self._sum_of_squares, self._min, self._max]
        ])

    def mean(self):
        return self._mu

    def var(self, unbiased=True):
        norm = self._num_samples
        if unbiased:
            norm = norm - 1
        return self._sum_of_squares / norm

    def var_of_mean(self):
        return self.var(unbiased=True) / self._num_samples

    def min(self):
        return self._min

    def max(self):
        return self._max

    def num_samples(self):
        return self._num_samples

    def __len__(self):
        return len(self._mu)


class MergerSummary(_SummaryStatistics):

    def __init__(
            self,
            sum_of_weights: Tensor, sum_of_squared_weights: Tensor,
            mu: Tensor, squared_deviation: Tensor, var_of_mean: Tensor,
            min: Tensor, max: Tensor):
        length = len(sum_of_weights)
        assert np.all([len(x) == length for x in [sum_of_squared_weights, mu, squared_deviation, var_of_mean, min, max]]), \
            '[ERROR] Data statistics mus all have the same number of entries'
        self._sum_of_weights = sum_of_weights
        self._sum_of_squared_weights = sum_of_squared_weights
        self._mu = mu
        self._squared_deviation = squared_deviation
        self._var_of_mean = var_of_mean
        self._min = min
        self._max = max

    def get_subset(self, index: Tensor):
        return MergerSummary(*[
            x[index]
            for x in [self._sum_of_weights, self._sum_of_squared_weights,
                      self._mu, self._squared_deviation, self._var_of_mean,
                      self._min, self._max]
        ])

    @classmethod
    def merge(cls, *weighted_summaries: Tuple['_SummaryStatistics', Union[Tensor, None]], index=None):
        if index is None:
            index = [torch.ones(len(s)) * i for i, (s, w) in enumerate(weighted_summaries)]
            index = torch.cat(index, dim=0).to(dtype=torch.long)
        num_nodes = np.sum([len(s) for s, w in weighted_summaries])
        assert len(index) == num_nodes
        mu = weighted_summaries[0][0].mean()
        sow, sosw, mu, sd, vom, min_, max_ = [torch.zeros(num_nodes, device=mu.device, dtype=mu.dtype) for _ in range(7)]
        for i, (s, w) in enumerate(weighted_summaries):
            mask = (index == i)
            assert torch.sum(mask) == len(s)
            if isinstance(s, MergerSummary):
                sow[mask] = s._sum_of_weights
                sosw[mask] = s._sum_of_squared_weights
                sd[mask] = s._squared_deviation
            elif isinstance(s, SampleSummary):
                sow[mask] = w
                sosw[mask] = w ** 2.
                sd[mask] = w * s.var(unbiased=False)
            else:
                raise Exception('[ERROR] Encountered unknown data summary')
            mu[mask] = s.mean()
            vom[mask] = s.var_of_mean()
            min_[mask] = s.min()
            max_[mask] = s.max()
        return cls(sow, sosw, mu, sd, vom, min_, max_)

    @classmethod
    def from_summaries(cls, stats1: _SummaryStatistics, weight1: Tensor, stats2: _SummaryStatistics, weight2: Tensor):
        return cls._from_summaries(stats1, weight1, stats2, weight2)

    @classmethod
    def _from_summaries(cls, stats1: _SummaryStatistics, weight1: Tensor, stats2: _SummaryStatistics, weight2: Tensor):
        sum_of_weights = weight1 + weight2
        sum_of_squared_weights = weight1 ** 2 + weight2 ** 2
        f = weight1 * weight2 / sum_of_weights
        mu1, mu2 = stats1.mean(), stats2.mean()
        delta = torch.abs(mu2 - mu1)
        mu = (mu1 * weight1 + mu2 * weight2) / sum_of_weights
        squared_deviation = weight1 * stats1.var(unbiased=False) + weight2 * stats2.var(unbiased=False) + f * delta ** 2
        var_of_mean = (weight1 ** 2 * stats1.var_of_mean() + weight2 ** 2 * stats1.var_of_mean()) / sum_of_weights ** 2
        min_ = torch.minimum(stats1.min(), stats2.min())
        max_ = torch.maximum(stats1.max(), stats2.max())
        return cls(sum_of_weights, sum_of_squared_weights, mu, squared_deviation, var_of_mean, min_, max_)

    def mean(self):
        return self._mu

    def var(self, unbiased=True):
        norm = self._sum_of_weights
        if unbiased:
            norm = norm - self._sum_of_squared_weights / self._sum_of_weights
        return self._squared_deviation / norm

    def num_samples(self):
        # effective sample size according to weights
        return self._sum_of_weights ** 2 / self._sum_of_squared_weights

    def var_of_mean(self):
        return self._var_of_mean

    def min(self):
        return self._min

    def max(self):
        return self._max

    def __len__(self):
        return len(self._mu)
