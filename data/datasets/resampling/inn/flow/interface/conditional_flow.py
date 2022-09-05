from typing import Any, Optional

from torch import nn, Tensor

from .flow_direction import FlowDirection
from .invertible_transform import InvertibleTransform
from ..log_likelihood import LogLikelihoodPerDimension


class ConditionalFlow(InvertibleTransform):

    def __init__(self, flow: InvertibleTransform, conditioner: nn.Module, num_condition_ports=1):
        super(ConditionalFlow, self).__init__({
            direction: flow.get_number_of_flow_ports(direction)
            for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]
        }, num_condition_ports=num_condition_ports)
        self.flow = flow
        self.conditioner = conditioner

    def forward_transform(self, *data: Any):
        flow_data, condition_data = self._process_condition_data(*data, direction=FlowDirection.FORWARD)
        return self.flow.forward_transform(*flow_data, *condition_data)

    def reverse_transform(self, *data: Any):
        flow_data, condition_data = self._process_condition_data(*data, direction=FlowDirection.REVERSE)
        return self.flow.reverse_transform(*flow_data, *condition_data)

    def _process_condition_data(self, *data: Any, direction: Optional[FlowDirection] = None):
        flow_data, condition_data = self.split_inputs(*data, direction=direction)
        condition_data = self.conditioner(*condition_data)
        if type(condition_data) == Tensor:
            condition_data = [condition_data]
        return flow_data, condition_data
    
    def add_determinant_tracker(self, ll_per_dim: LogLikelihoodPerDimension):
        self.flow.add_determinant_tracker(ll_per_dim)
        return super(ConditionalFlow, self).add_determinant_tracker(ll_per_dim)