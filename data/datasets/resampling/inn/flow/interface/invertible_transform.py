from typing import Mapping, Optional, Dict, List, Tuple, Any
import torch.nn as nn

from .flow_direction import FlowDirection
from ..log_likelihood import LogLikelihoodPerDimension


class InvertibleTransform(nn.Module):

    def __init__(self, num_flow_ports: Mapping[FlowDirection, int], num_condition_ports: Optional[int] = 0, label: Optional[str] = None):
        super(InvertibleTransform, self).__init__()
        self.label = label
        self.__num_flow_ports: Dict[FlowDirection, int] = {
            direction: num_flow_ports[direction] if direction in num_flow_ports else 0
            for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]
        }
        self.__num_condition_ports = num_condition_ports
        self.__transform_mapping = {
            FlowDirection.FORWARD: self.forward_transform,
            FlowDirection.REVERSE: self.reverse_transform,
        }
        self.determinant_trackers: List[LogLikelihoodPerDimension] = []

    def is_labeled(self):
        return (self.label is not None)

    def set_num_flow_ports(self, num_ports: int, direction: FlowDirection):
        self.__num_flow_ports.update({direction: num_ports})
        return self

    def set_num_condition_ports(self, num_ports: int):
        self.__num_condition_ports = num_ports
        return self

    def get_input_length(self, direction: Optional[FlowDirection] = FlowDirection.FORWARD) -> int:
        return self.__num_flow_ports[direction] + self.__num_condition_ports

    def get_output_length(self, direction: Optional[FlowDirection] = FlowDirection.FORWARD) -> int:
        return self.__num_flow_ports[direction.opposite()]

    def get_number_of_condition_ports(self) -> int:
        return self.__num_condition_ports

    def get_number_of_flow_ports(self, direction: FlowDirection) -> int:
        return self.__num_flow_ports[direction]

    def forward(self, *data: Any, direction: Optional[FlowDirection] = FlowDirection.FORWARD):
        return self.__transform_mapping[direction](*data)

    def forward_transform(self, *data: Any) -> Tuple[Any, ...]:
        raise NotImplementedError()

    def reverse_transform(self, *data: Any) -> Tuple[Any, ...]:
        raise NotImplementedError()

    def add_determinant_tracker(self, ll_per_dim: LogLikelihoodPerDimension):
        self.determinant_trackers.append(ll_per_dim)

    def update_determinant_trackers(self, log_abs_det_per_dim: Any, dimensions: int):
        for ll_tracker in self.determinant_trackers:
            ll_tracker.add_flow_log_determinant(log_abs_det_per_dim, dimensions)

    def clear_determinant_trackers(self):
        self.determinant_trackers = []

    def has_determinant_tracker(self):
        return len(self.determinant_trackers) > 0

    def split_inputs(self, *data: Any, direction: FlowDirection):
        num_flow_ports = self.get_number_of_flow_ports(direction)
        num_condition_ports = self.get_number_of_condition_ports()
        if len(data) != num_flow_ports + num_condition_ports:
            raise Exception('[ERROR] Something went wrong while splitting inputs...')
        flow_data = data[:num_flow_ports]
        condition_data = data[num_flow_ports:]
        return flow_data, condition_data
