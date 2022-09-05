from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from data.datasets import OutputMode
from .backend_output_mode import BackendOutputMode
from .constant_channel_parameterization import ConstantChannelParameterization


class _RGBOOutput(ConstantChannelParameterization):

    def backend_output_mode(self) -> BackendOutputMode:
        raise NotImplementedError()

    def __init__(self):
        super(_RGBOOutput, self).__init__(4, OutputMode.RGBO)

    def split_inputs(self, network_output: Tensor):
        return network_output[:, :3], network_output[:, 3:]

    def join_inputs(self, rgb: Tensor, o: Tensor):
        return torch.cat([rgb, o],dim=-1)

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        rgb, o = self.split_inputs(network_output)
        rgb, o = self.transform_rgbo_for_rendering(rgb, o)
        return self.join_inputs(rgb, o)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        rgb, o = self.split_inputs(network_output)
        rgb, o = self.transform_rgbo_for_training(rgb, o)
        return self.join_inputs(rgb, o)

    def transform_rgbo_for_rendering(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()

    def transform_rgbo_for_training(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError()


class SoftClampRGBOOutput(_RGBOOutput):

    @staticmethod
    def _transform(rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.sigmoid(rgb), F.softplus(o)

    def transform_rgbo_for_rendering(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        raise self._transform(rgb, o)

    def transform_rgbo_for_training(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        raise self._transform(rgb, o)

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.RGBO


class DirectRGBOOutput(_RGBOOutput):

    def transform_rgbo_for_rendering(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.clamp(rgb, min=0., max=1.), torch.clamp(o, min=0.)

    def transform_rgbo_for_training(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        return rgb, o

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.RGBO_DIRECT


class SoftClampExponentialRGBOOutput(_RGBOOutput):

    @staticmethod
    def _transform(rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        return torch.sigmoid(rgb), torch.exp(o)

    def transform_rgbo_for_rendering(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        raise self._transform(rgb, o)

    def transform_rgbo_for_training(self, rgb: Tensor, o: Tensor) -> Tuple[Tensor, Tensor]:
        raise self._transform(rgb, o)

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.RGBO_EXP


CHOICES = {
    '': SoftClampRGBOOutput,
    'soft-clamp': SoftClampRGBOOutput,
    'direct': DirectRGBOOutput,
    'soft-clamp-exp': SoftClampExponentialRGBOOutput
}
