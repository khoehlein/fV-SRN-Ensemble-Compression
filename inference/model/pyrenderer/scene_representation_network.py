import argparse
from typing import Dict, Any, Optional, Union, Tuple, List

import pyrenderer
from torch import Tensor

from data.datasets import OutputMode
from data.datasets.output_mode import MultivariateOutputMode
from .input_parameterization import PyrendererInputParameterization
from .latent_features import PyrendererLatentFeatures
from .core_network import PyrendererCoreNetwork
from .output_parameterization import PyrendererOutputParameterization
from ..scene_representation_network import ModularSRN
from ..scene_representation_network.evaluation_mode import EvaluationMode


class PyrendererSRN(ModularSRN):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser, output_mode: Optional[OutputMode] = None):
        PyrendererInputParameterization.init_parser(parser)
        PyrendererLatentFeatures.init_parser(parser)
        PyrendererOutputParameterization.init_parser(parser, output_mode=output_mode)
        PyrendererCoreNetwork.init_parser(parser)

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            member_keys: Optional[List[int]] = None, dataset_key_times: Optional[List[float]] = None,
            output_mode: Optional[Union[OutputMode, MultivariateOutputMode]] = None
    ):
        input_parameterization = PyrendererInputParameterization.from_dict(args)
        latent_features = PyrendererLatentFeatures.from_dict(args, member_keys=member_keys, dataset_key_times=dataset_key_times)
        output_parameterization = PyrendererOutputParameterization.from_dict(args, output_mode=output_mode)
        core_network = PyrendererCoreNetwork.from_dict(args, input_parameterization, latent_features, output_parameterization, member_keys=member_keys)
        return cls(input_parameterization, core_network, output_parameterization, latent_features=latent_features)

    def export_to_pyrenderer(
            self,
            grid_encoding, return_grid_encoding_error=False,
            network: Optional[pyrenderer.SceneNetwork] = None,
            time = None,
            ensemble = None
    ) -> Union[pyrenderer.SceneNetwork, Tuple[pyrenderer.SceneNetwork, float]]:
        if self.input_parameterization.uses_time() and (
                self.latent_features is None or not self.latent_features.uses_time()):
            raise RuntimeError(
                "[ERROR] Time input for pyrenderer.SceneNetwork() works only for time-dependent latent grids (for now).")
        network = self.input_parameterization.export_to_pyrenderer(network=network)
        network = self.output_parameterization.export_to_pyrenderer(network=network)
        padding = 0
        if self.latent_features is not None:
            network, error, padding = self.latent_features.export_to_pyrenderer(
                grid_encoding, network, time=time, ensemble=ensemble)
        else:
            error = 0.
        network = self.core_network.export_to_pyrenderer(
            network=network, time=time, ensemble=ensemble, pad_first_layer=padding)
        if not network.valid():
            raise RuntimeError('[ERROR] Failed to convert scene representation network to tensor cores.')
        if return_grid_encoding_error:
            return network, error
        return network

    def forward(
            self,
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member:Tensor,
            evaluation_mode: Union[EvaluationMode, str]
    ) -> Tensor:
        if type(evaluation_mode) == str:
            if evaluation_mode == 'world':
                evaluation_mode = EvaluationMode.TRAINING
            elif evaluation_mode == 'screen':
                evaluation_mode = EvaluationMode.RENDERING
            else:
                raise NotImplementedError()
        return super(PyrendererSRN, self).forward(positions, transfer_functions, time, member, evaluation_mode)

    def use_direction(self) -> bool:
        return self.uses_direction()

    def start_epoch(self) -> bool:
        return False

    def base_input_channels(self):
        return 3
