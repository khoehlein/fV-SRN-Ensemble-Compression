from typing import Optional

import pyrenderer
from torch import Tensor, nn

from data.datasets import OutputMode
from .backend_output_mode import BackendOutputMode
from ..scene_representation_network.evaluation_mode import EvaluationMode


class IOutputParameterization(nn.Module):

    def input_channels(self) -> int:
        raise NotImplementedError()

    def output_channels(self) -> int:
        raise NotImplementedError()

    def output_mode(self) -> OutputMode:
        raise NotImplementedError()

    def backend_output_mode(self) -> BackendOutputMode:
        raise NotImplementedError()

    def forward(self, network_output: Tensor, evaluation_mode: EvaluationMode) -> Tensor:
        raise NotImplementedError()

    def export_to_pyrenderer(self, network: Optional[pyrenderer.SceneNetwork] = None):
        if network is None:
            network = pyrenderer.SceneNetwork()
        backend_output_mode = self.backend_output_mode()
        converter = pyrenderer.SceneNetwork.OutputParametrization.OutputModeFromString
        network.output.output_mode = converter(backend_output_mode.value)
        return network
