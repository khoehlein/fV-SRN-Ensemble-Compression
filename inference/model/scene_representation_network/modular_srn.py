from typing import Optional

from torch import Tensor

from ..input_parameterization import IInputParameterization
from ..latent_features import ILatentFeatures
from ..core_network import ICoreNetwork
from ..output_parameterization import IOutputParameterization
from .interface import ISceneRepresentationNetwork, EvaluationMode


class ModularSRN(ISceneRepresentationNetwork):

    def __init__(
            self,
            input_parameterization: IInputParameterization,
            core_network: ICoreNetwork,
            output_parameterization: IOutputParameterization,
            latent_features: Optional[ILatentFeatures] = None
    ):
        super(ModularSRN, self).__init__()
        self.input_parameterization = input_parameterization
        self.core_network = core_network
        self.output_parameterization = output_parameterization
        self.latent_features = latent_features

    @property
    def _device(self):
        return next(self.parameters()).data.device

    def forward(
            self,
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor,
            evaluation_mode: EvaluationMode
    ) -> Tensor:
        data_input = self.input_parameterization.forward(positions, transfer_functions, time, member)
        if self.latent_features is not None:
            latent_inputs = self.latent_features.evaluate(positions, time, member)
        else:
            latent_inputs = None
        network_output = self.core_network.forward(data_input, latent_inputs, positions, transfer_functions, time, member)
        prediction = self.output_parameterization.forward(network_output, evaluation_mode)
        return prediction

    def uses_positions(self):
        return self.input_parameterization.uses_positions() or self.latent_features.uses_positions()

    def uses_direction(self):
        return self.input_parameterization.uses_directions()

    def uses_time(self):
        return self.input_parameterization.uses_time() or self.latent_features.uses_time()

    def uses_member(self):
        return self.input_parameterization.uses_member() or \
               self.latent_features.uses_member() or \
               self.core_network.uses_member()

    def num_members(self):
        if self.latent_features.uses_member():
            return self.latent_features.num_members()
        if self.core_network.uses_member():
            return self.core_network.num_members()
        return 1

    def uses_transfer_functions(self):
        return False

    def backend_output_mode(self):
        return self.output_parameterization.backend_output_mode()

    def output_mode(self):
        return self.output_parameterization.output_mode()

    def output_channels(self):
        return self.output_parameterization.output_channels()
