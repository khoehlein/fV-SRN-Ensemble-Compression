import torch
from torch import Tensor

from data.datasets import OutputMode
from .constant_channel_parameterization import ConstantChannelParameterization
from .backend_output_mode import BackendOutputMode



class _DensityOutput(ConstantChannelParameterization):

    def __init__(self):
        super(_DensityOutput, self).__init__(1, OutputMode.DENSITY)


class SoftClampDensityOutput(_DensityOutput):

    @staticmethod
    def _transform(network_output: Tensor) -> Tensor:
        return torch.sigmoid(network_output)

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return self._transform(network_output)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return self._transform(network_output)

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY


class DirectDensityOutput(_DensityOutput):

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output # torch.clamp(network_output, min=0, max=1)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY_DIRECT


class MixedClampDensityOutput(_DensityOutput):

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return torch.clamp(network_output, min=0, max=1)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY_DIRECT


class HardClampDensityOutput(_DensityOutput):

    @staticmethod
    def _transform(network_output: Tensor) -> Tensor:
        return torch.clamp(network_output, min=0, max=1)

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return self._transform(network_output)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return self._transform(network_output)

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY


CHOICES = {
    '': SoftClampDensityOutput,
    'soft-clamp': SoftClampDensityOutput,
    'hard-clamp': HardClampDensityOutput,
    'mixed': MixedClampDensityOutput,
    'direct': DirectDensityOutput,
}