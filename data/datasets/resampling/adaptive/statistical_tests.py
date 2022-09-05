import math

import numpy as np
import torch


class FastWhiteHomoscedasticityTest(object):

    def __init__(self, alpha=0.05, simplify_predictors=True):
        self.alpha = alpha
        self.simplify_predictors = simplify_predictors

    def compute(self, samples: torch.Tensor, coordinates: torch.Tensor):
        coordinates = torch.transpose(coordinates, 1, 0) # coordinates shape: (nodes, num_samples, dim)
        samples = torch.transpose(samples, 1, 0) # samples shape (nodes, num_samples)
        X = self._add_constant(coordinates)
        _, predictions, residuals = self._ols(X, samples)
        if self.simplify_predictors:
            X = self._add_constant(predictions[..., None])
        i0, i1 = np.triu_indices(X.shape[-1])
        X = X[..., i0] * X[..., i1]
        residuals_squared = residuals ** 2
        _, predictions, residuals = self._ols(X, residuals_squared)
        r_squared = self._compute_r_squared(residuals_squared, residuals)
        lm = samples.shape[-1] * r_squared
        df = X.shape[-1] - 1 # degrees of freedom
        p_value = self._chi2_sf(lm, df)
        test_statistic = lm
        return FastWhiteHomoscedasticityTest.Result(test_statistic, p_value, self.alpha)

    @staticmethod
    def _add_constant(data: torch.Tensor):
        constant = torch.ones(data.shape[:-1], dtype=data.dtype, device=data.device)
        X = torch.cat([constant[..., None], data], dim=-1)
        return X

    @staticmethod
    def _ols(X: torch.Tensor, targets: torch.Tensor):
        pinv = torch.linalg.pinv(X)  # shape (nodes, dim + 1, num_samples)
        weights = torch.bmm(pinv, targets[..., None])  # shape (*nodes, dim+ 1, 1)
        predictions = torch.bmm(X, weights)[..., 0]  # shape (*nodes, num_samples)
        residuals = targets - predictions
        return weights, predictions, residuals

    @staticmethod
    def _compute_r_squared(samples, residuals):
        ss_res = torch.mean(residuals ** 2, dim=-1)
        ss_tot = torch.var(samples, dim=-1, unbiased=False)
        return 1. - (ss_res / ss_tot)

    @staticmethod
    def _chi2_sf(x, df):
        return torch.igammac(torch.full_like(x, df /2.), x / 2.)

    class Result(object):

        def __init__(self, test_tatistic, p_value, alpha):
            self.test_tatistic = test_tatistic
            self.p_value = p_value
            self.alpha = alpha

        def reject(self):
            return self.p_value < self.alpha


class FastKolmogorovSmirnovTestNd(object):

    def __init__(self, alpha=0.05):
        assert alpha > 0
        self.alpha = alpha

    def critical_factor(self):
        return math.sqrt(-math.log(self.alpha / 2.) / 2.)

    def threshold(self, n1, n2):
        return self.critical_factor() * torch.sqrt((n1 + n2) / (n1 * n2))

    def compute(self, samples: torch.Tensor, classification: torch.Tensor):
        assert classification.shape[:-1] == samples.shape, f'[ERROR] Expected shape {samples.shape},got {classification.shape[:-1]} instead.'
        sample_order = torch.argsort(samples, dim=0)

        leading_dim_grid = [ax.reshape(-1) for ax in torch.meshgrid(*[torch.arange(s) for s in samples.shape])[1:]]
        ordered_grid = (sample_order.reshape(-1), *leading_dim_grid)
        ordered_classification = torch.reshape(classification[ordered_grid], classification.shape)
        sample_counts = torch.stack([
            torch.cumsum(ordered_classification.to(dtype=torch.int), dim=0),
            torch.cumsum((~ordered_classification).to(dtype=torch.int), dim=0)
        ], dim=0) # shape (left/right, num_samples, nodes, dim)
        total_samples = sample_counts[:, -1, ...] # shape (l/r, nodes, dim)
        assert torch.all(total_samples) > 0
        cdfs = sample_counts / total_samples[:, None, ...]
        test_statistics = torch.amax(torch.abs(torch.diff(cdfs, dim=0)[0]), dim=0) # shape (nodes, dim)
        thresholds = self.threshold(total_samples[0], total_samples[1])
        significance_ratios = test_statistics / thresholds
        return FastKolmogorovSmirnovTestNd.Result(test_statistics, significance_ratios)

    class Result(object):

        def __init__(self, test_statistics, significance_ratios):
            self.test_statistics = test_statistics
            self.significance_ratios = significance_ratios

        def reject(self):
            return torch.any(self.significance_ratios > 1., dim=-1)

        def best_split(self):
            return torch.argmax(self.significance_ratios, dim=-1)

        def split_ranks(self):
            return torch.argsort(torch.argsort(self.significance_ratios, dim=-1, descending=True), dim=-1)