from enum import Enum

import numpy as np
from torch import nn

from analysis.data_ssim import DSSIM2d
from data.necker_ensemble.single_variable import revert_scaling
from inference.model.pyrenderer import PyrendererSRN


class CompressionStatistics(object):

    def __init__(self, network: PyrendererSRN):
        self.core_parameters = self._get_parameter_count(network.core_network)
        self.grid_parameters = self._get_parameter_count(network.latent_features)
        self.num_members = network.num_members()

    def _get_parameter_count(self, m: nn.Module):
        return sum([p.numel() for p in m.parameters()])

    def byte_size(self):
        return self.core_parameters * 2 + self.grid_parameters

    def compression_rate(self, resolution=None):
        if resolution is None:
            resolution = (352, 250, 12)
        return int(np.prod(resolution) * self.num_members) * 4 / self.byte_size()


class DeviationStatistics(object):

    class Measure(Enum):
        MSE = 'mse'
        RMSE = 'rmse'
        MAE = 'mae'
        MEDAE = 'medae'
        MAXAE = 'maxae'
        PSNR = 'psnr'
        DSSIM = 'dssim'

    def __init__(self, data: np.ndarray, restored: np.ndarray, scales=None):
        self._measures_rescaled = self.compute_deviation_measures(data, restored)
        if scales is not None:
            data_r = revert_scaling(data, scales)
            restored_r = revert_scaling(restored, scales)
            self._measures_reverted = self.compute_deviation_measures(data_r, restored_r)
        else:
            self._measures_reverted = {}

    def compute_deviation_measures(self, data, restored):
        deviation = np.abs(data - restored)
        mse = np.mean(deviation ** 2)
        range = self.compute_range(data, restored)
        return {
            DeviationStatistics.Measure.MSE: mse,
            DeviationStatistics.Measure.RMSE: np.sqrt(mse),
            DeviationStatistics.Measure.MAE: np.mean(deviation),
            DeviationStatistics.Measure.MEDAE: np.median(deviation),
            DeviationStatistics.Measure.MAXAE: np.max(deviation),
            DeviationStatistics.Measure.PSNR: 10 * (np.log(range) - np.log10(mse)),
            DeviationStatistics.Measure.DSSIM: self.compute_dssim(data, restored),
        }

    def compute_range(self, *data):
        min_val = min([np.min(x) for x in data])
        max_val = max([np.max(x) for x in data])
        return max_val - min_val

    def compute_dssim(self, data, restored):
        data = np.transpose(data, (0, 3, 2, 1))
        restored = np.transpose(restored, (0, 3, 2, 1))
        assert data.shape[1:] == (12, 352, 250)
        assert restored.shape == data.shape
        dssim = DSSIM2d()
        values = dssim(data, restored)
        out = np.mean(np.min(values, axis=-1))
        return out

    def to_dict(self):
        return {
            **{key.value + '_rescaled': self._measures_rescaled[key] for key in self._measures_rescaled},
            **{key.value + '_reverted': self._measures_reverted[key] for key in self._measures_reverted},
        }