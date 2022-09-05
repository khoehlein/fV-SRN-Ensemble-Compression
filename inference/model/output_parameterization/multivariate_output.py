import torch
from torch import Tensor

from data.datasets import OutputMode
from inference.model.output_parameterization import BackendOutputMode
from inference.model.output_parameterization.constant_channel_parameterization import ConstantChannelParameterization


class MultivariateOutput(ConstantChannelParameterization):

    def __init__(self, channels, active_output=0):
        super(MultivariateOutput, self).__init__(channels, OutputMode.MULTIVARIATE)
        self.active_output = active_output

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output[:, [self.active_output]]

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY


class MultivariateClampedOutput(MultivariateOutput):

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return torch.clip(network_output[:, [self.active_output]], min=0., max=1.)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output
