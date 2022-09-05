from torch import Tensor

from .scene_representation_network import ISceneRepresentationNetwork
from ..field_evaluator import IFieldEvaluator


class NetworkEvaluator(IFieldEvaluator):

    def __init__(self, network: ISceneRepresentationNetwork):
        super(NetworkEvaluator, self).__init__(network.base_input_channels(), network.output_channels(), network._device)
        self.network = network

    def forward(self, positions: Tensor) -> Tensor:
        raise NotImplementedError()
