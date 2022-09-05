from torch import nn, Tensor

from .evaluation_mode import EvaluationMode


class ISceneRepresentationNetwork(nn.Module):

    # def generalize_to_new_ensembles(self, num_members: int):
    #     raise NotImplementedError()

    # def supports_mixed_latent_spaces(self):
    #     raise NotImplementedError()

    def uses_direction(self):
        raise NotImplementedError()

    def uses_time(self):
        raise NotImplementedError()

    def uses_positions(self):
        raise NotImplementedError()

    def uses_member(self):
        raise NotImplementedError()

    def num_members(self):
        raise NotImplementedError()

    def uses_transfer_functions(self):
        raise NotImplementedError()

    def backend_output_mode(self):
        raise NotImplementedError()

    def output_mode(self):
        raise NotImplementedError()

    # def start_epoch(self):
    #     raise NotImplementedError()

    def forward(
            self,
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member:Tensor,
            evaluation_mode: EvaluationMode
    ):
        raise NotImplementedError()

    def output_channels(self):
        raise NotImplementedError()