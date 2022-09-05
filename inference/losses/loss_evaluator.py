from typing import Optional

from torch import Tensor

from inference import IFieldEvaluator
from inference.model import NetworkEvaluator
from inference.volume import VolumeEvaluator


class LossEvaluator(IFieldEvaluator):

    def __init__(self, loss_function, volume: Optional[VolumeEvaluator] = None, network: Optional[NetworkEvaluator] = None, dimension=None, device=None):
        self.network_evaluator = None
        self.volume_evaluator = None
        network, volume = self._verify_sources(network=network, volume=volume)
        if volume is not None:
            in_dimension = volume.in_channels()
            device_ = volume.device
        elif network is not None:
            in_dimension = network.in_channels()
            device_ = network.device
        else:
            in_dimension = dimension
            device_ = device
        assert in_dimension is not None
        assert dimension is None or dimension == in_dimension
        assert device_ is not None
        assert device is None or device == device_
        super(LossEvaluator, self).__init__(in_dimension, 1, device_)
        self.network_evaluator = network
        self.volume_evaluator = volume
        self.loss_function = loss_function

    def forward(self, positions: Tensor) -> Tensor:
        volume_data = self.volume_evaluator.evaluate(positions)
        network_data = self.network_evaluator.evaluate(positions)
        return self.loss_function(volume_data, network_data, reduction='none')

    def _verify_sources(self, network: Optional[NetworkEvaluator] = None, volume: Optional[VolumeEvaluator] = None):
        if network is None:
            network = self.network_evaluator
        if volume is None:
            volume = self.volume_evaluator
        if volume is not None and network is not None:
            assert volume.in_channels() == network.in_channels()
            assert volume.out_channels() == network.out_channels()
            assert volume.device == network.device
        return network, volume

    def set_source(self, network: Optional[NetworkEvaluator] = None, volume: Optional[VolumeEvaluator] = None, volume_data=None):
        self.network_evaluator, self.volume_evaluator = self._verify_sources(network=network, volume=volume)
        if volume_data is not None:
            assert  self.volume_evaluator is not None, '[ERROR] Trying to update volume data source, but volume evaluator was not given.'
            self.volume_evaluator.set_source(volume_data)
        return self
