from itertools import chain
from typing import Dict, Optional, Any, List, Union

from torch import nn
from .flow_direction import FlowDirection
from .invertible_transform import InvertibleTransform
from ..log_likelihood import LogLikelihoodPerDimension


class FlowPort(object):

    def __init__(self, label: Optional[str] = None, except_on_flush_none: Optional[bool] = True):
        self.label = label
        self.data = None
        self.except_on_flush_none = except_on_flush_none

    def carries_data(self):
        return self.data is not None

    def set_data(self, data: Any):
        self.data = data
        return self

    def flush(self):
        data = self.data
        if data is None and self.except_on_flush_none:
            raise Exception('[ERROR] Trying to flush None.')
        self.data = None
        return data


class PortRegister(object):

    def __init__(self):
        self.__ports: List[FlowPort] = []
        self.__port_index: Dict[FlowPort, int] = {}

    def get_port_list(self):
        return self.__ports

    def register_ports(self, *ports: FlowPort):
        num_ports = len(self.__ports)
        self.__ports += list(ports)
        self.__port_index.update({port: i for port, i in zip(ports, range(num_ports, num_ports + len(ports)))})
        return self

    def delete_ports(self, *ports: FlowPort):
        self.__ports = [port for port in self.__ports if port not in ports]
        self.__port_index = {port: i for i, port in enumerate(self.__ports)}
        return self

    def __contains__(self, item):
        return item in self.__port_index

    def __len__(self):
        return len(self.__ports)


class FlowInterface(object):

    def __init__(self, transform: Union[InvertibleTransform, None]):
        self.transform = transform
        self.__flow_ports: Dict[FlowDirection, PortRegister] = {
            direction: PortRegister()
            for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]
        }
        self.__condition_ports = PortRegister()
        if transform is not None:
            for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]:
                self.__flow_ports[direction].register_ports(
                    *[FlowPort() for _ in range(transform.get_number_of_flow_ports(direction))]
                )
            self.__condition_ports.register_ports(
                *[FlowPort() for _ in range(transform.get_number_of_condition_ports())]
            )

    def create_empty_flow_ports(self, num_ports: int, direction: FlowDirection):
        new_ports = [FlowPort() for _ in range(num_ports)]
        self.__flow_ports[direction].register_ports(*new_ports)
        return new_ports

    def create_empty_condition_ports(self, num_ports: int):
        new_ports = [FlowPort() for _ in range(num_ports)]
        self.__condition_ports.register_ports(*new_ports)
        return new_ports

    def delete_empty_flow_ports(self, *ports):
        for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]:
            register = self.__flow_ports[direction]
            register.delete_ports(*[port for port in ports if port in register])

    def get_input_ports(self,direction: FlowDirection):
        return self.__flow_ports[direction].get_port_list() + self.__condition_ports.get_port_list()

    def get_output_ports(self, direction: FlowDirection):
        return self.__flow_ports[direction.opposite()].get_port_list()

    def get_flow_ports(self, direction: FlowDirection):
        return self.__flow_ports[direction].get_port_list()

    def get_condition_ports(self):
        return self.__condition_ports.get_port_list()

    @staticmethod
    def _ports_in_list_ready(port_list: List[FlowPort]):
        for port in port_list:
            if not port.carries_data():
                return False
        return True

    def all_inputs_ready(self, direction: FlowDirection):
        return self._ports_in_list_ready(self.get_input_ports(direction))

    def flush_inputs(self, direction: FlowDirection):
        return [port.flush() for port in self.get_input_ports(direction)]

    def flush_outputs(self, direction):
        return [port.flush() for port in self.get_output_ports(direction)]

    def set_inputs(self, *data: Any, direction: FlowDirection = None):
        assert direction is not None, '[ERROR] Direction must be provided'
        in_ports = self.get_input_ports(direction)
        assert len(data) == len(in_ports), \
            '[ERROR] Number of provided data items must match the number of input ports.'
        for port, item in zip(in_ports, data):
            port.set_data(item)
        return self

    def set_outputs(self, *data: Any, direction: FlowDirection = None):
        assert direction is not None, '[ERROR] Direction must be provided'
        out_ports = self.get_output_ports(direction)
        if len(data) != len(out_ports):
            raise Exception('[ERROR] Number of provided data items must match the number of output ports.')
        for port, item in zip(out_ports, data):
            port.set_data(item)
        return out_ports

    def call_transform(self, direction: FlowDirection):
        if self.transform is None:
            raise Exception('[ERROR] Cannot call transform if transform is None.')
        data = self.flush_inputs(direction)
        outputs = self.transform.forward(*data, direction=direction)
        return self.set_outputs(*outputs, direction=direction)

    def is_condition_port(self, port: FlowPort):
        return (port in self.__condition_ports)


class NormalizingFlow(InvertibleTransform):

    def __init__(self):
        super(NormalizingFlow, self).__init__({FlowDirection.FORWARD: 0, FlowDirection.REVERSE: 0})
        self.__interface = FlowInterface(None)
        self.__transform_modules = nn.ModuleList([])
        self.__transform_modules_by_label: Dict[str, InvertibleTransform] = {}
        self.__transform_interfaces_by_label: Dict[str, FlowInterface] = {}
        self.__transform_interfaces_by_port: Dict[FlowPort, FlowInterface] = {}
        self.__transform_interfaces_by_transform: Dict[InvertibleTransform, FlowInterface] = {}
        self.__flow_port_connections: Dict[FlowDirection, Dict[FlowPort, FlowPort]] = {
            direction: {} for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]
        }
        self.__condition_port_connections: Dict[FlowPort, List[FlowPort]] = {}
        self.__flow_ports_by_label: Dict[str, FlowPort] = {}
        self.__condition_ports_by_label: Dict[str, FlowPort] = {}

    def add_flow_port(self, direction: FlowDirection, label: Optional[str] = None):
        port = self._create_empty_flow_ports(1, direction)[0]
        if label is not None:
            if label in self.__flow_ports_by_label:
                raise Exception(f'[ERROR] Label {label} is already used by another flow port.')
            self.__flow_ports_by_label[label] = port
        counter_port = self._create_empty_flow_ports(1, direction.opposite())[0]
        self.connect_flow_ports(port, counter_port, direction)
        return port

    def add_condition_port(self, label: Optional[str] = None):
        port = self._create_empty_condition_ports(1)[0]
        if label is not None:
            if label in self.__condition_ports_by_label:
                raise Exception(f'[ERROR] Label {label} is already used by another condition port.')
            self.__condition_ports_by_label[label] = port
        return port

    def connect_flow_ports(self, first_port, second_port, direction):
        if first_port in self.__flow_port_connections[direction]:
            self._handle_existing_connection(first_port, second_port, direction)
        if second_port in self.__flow_port_connections[direction.opposite()]:
            self._handle_existing_connection(second_port, first_port, direction.opposite())
        self.__flow_port_connections[direction][first_port] = second_port
        self.__flow_port_connections[direction.opposite()][second_port] = first_port

    def connect_condition_ports(self, source: FlowPort, sink: FlowPort):
        if not self._is_condition_source(source):
            raise Exception('[ERROR] {source} is not a valid condition source port.')
        self.__condition_port_connections[source].append(sink)

    def _is_condition_source(self, port: FlowPort):
        return port in self.__condition_port_connections

    def _create_empty_flow_ports(self, num_ports: int, direction: FlowDirection):
        ports = self.__interface.create_empty_flow_ports(num_ports, direction)
        self._update_interface_by_port_register(self.__interface, *ports)
        self._reset_port_counts()
        return ports

    def _delete_empty_ports(self, *ports):
        self.__interface.delete_empty_flow_ports(*ports)
        for port in ports:
            del self.__transform_interfaces_by_port[port]
            for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]:
                if port in self.__flow_port_connections[direction]:
                    del self.__flow_port_connections[direction][port]
        self._reset_port_counts()

    def _create_empty_condition_ports(self, num_ports: int):
        ports = self.__interface.create_empty_condition_ports(num_ports)
        self._update_interface_by_port_register(self.__interface, *ports)
        self.__condition_port_connections.update({port: [] for port in ports})
        self._reset_port_counts()
        return ports

    def _update_interface_by_port_register(self, interface, *ports):
        self.__transform_interfaces_by_port.update({port: interface for port in ports})

    def _reset_port_counts(self):
        for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]:
            total_num_ports = len(self.__interface.get_flow_ports(direction))
            self.set_num_flow_ports(total_num_ports, direction)
        total_num_ports = len(self.__interface.get_condition_ports())
        self.set_num_condition_ports(total_num_ports)

    def _handle_existing_connection(self, port, new_connection, direction):
        old_connection = self.__flow_port_connections[direction][port]
        if old_connection is new_connection:
            return
        interface_of_old_connection = self.__transform_interfaces_by_port[old_connection]
        if interface_of_old_connection is self.__interface:
            self._delete_empty_ports(old_connection)
        else:
            replacement_port = self._create_empty_flow_ports(1, direction)[0]
            self.__flow_port_connections[direction][replacement_port] = old_connection
            self.__flow_port_connections[direction.opposite()][old_connection] = replacement_port

    def add_transform(
            self,
            transform: InvertibleTransform,
            inputs: Optional[List[FlowPort]] = None,
            direction: Optional[FlowDirection] = FlowDirection.FORWARD,
            conditions: Optional[Union[FlowPort, List[FlowPort]]] = None,
            return_transform: Optional[bool] = False,
    ):
        self._register_transform_module(transform)
        for determinant_tracker in self.determinant_trackers:
            transform.add_determinant_tracker(determinant_tracker)
        transform_interface = self._create_transform_interface(transform)
        if inputs is not None:
            if type(inputs) == FlowPort:
                inputs = [inputs]
            self._update_transform_inputs(transform_interface, inputs, direction)
        output_ports = transform_interface.get_output_ports(direction)
        if conditions is not None:
            if type(conditions) == FlowPort:
                conditions =[conditions]
            self._update_transform_conditions(transform_interface, conditions)
        if return_transform:
            return output_ports, transform
        else:
            return output_ports

    def _register_transform_module(self, transform: InvertibleTransform):
        self.__transform_modules.append(transform)
        if transform.is_labeled():
            label = transform.label
            if label in self.__transform_modules_by_label:
                raise Exception(f'[ERROR] Transform label {label} is already used by another transform.')
            else:
                self.__transform_modules_by_label[label] = transform

    def _create_transform_interface(self, transform: InvertibleTransform):
        transform_interface = FlowInterface(transform)
        self._register_transform_flow_ports(transform_interface, FlowDirection.FORWARD)
        self._register_transform_flow_ports(transform_interface, FlowDirection.REVERSE)
        self._register_transform_condition_ports(transform_interface)
        self.__transform_interfaces_by_transform[transform] = transform_interface
        if transform.is_labeled():
            self.__transform_interfaces_by_label[transform.label] = transform_interface
        return transform_interface

    def _update_transform_inputs(self, transform_interface, inputs, direction):
        input_ports = transform_interface.get_flow_ports(direction)
        if len(input_ports) != len(inputs):
            raise Exception('[ERROR] Number of input ports must match the number of required inputs.')
        for module_port, input_port in zip(input_ports, inputs):
            self.connect_flow_ports(input_port, module_port, direction)

    def _update_transform_conditions(self, transform_interface, conditions):
        condition_ports = transform_interface.get_condition_ports()
        if len(condition_ports) != len(conditions):
            raise Exception('[ERROR] Number of condition ports must match the number of required conditions.')
        for module_port, condition_port in zip(condition_ports, conditions):
            self.connect_condition_ports(condition_port, module_port)

    def _register_transform_flow_ports(self, transform_interface, direction):
        transform_ports = transform_interface.get_flow_ports(direction)
        self._update_interface_by_port_register(transform_interface, *transform_ports)
        ports = self._create_empty_flow_ports(len(transform_ports), direction)
        for module_port, transform_port in zip(ports, transform_ports):
            self.connect_flow_ports(module_port, transform_port, direction)

    def _register_transform_condition_ports(self, transform_interface: FlowInterface):
        transform_ports = transform_interface.get_condition_ports()
        self._update_interface_by_port_register(transform_interface, *transform_ports)

    def forward(self,*data, direction: FlowDirection = FlowDirection.FORWARD):
        self.__interface.set_inputs(*data, direction=direction)
        changed_ports = self.__interface.get_input_ports(direction)
        while len(changed_ports) > 0:
            changed_ports = self._propagate_port_data(*changed_ports, direction=direction)
        outputs = self.__interface.flush_outputs(direction)
        return outputs

    def _propagate_port_data(self, *ports: FlowPort, direction: Optional[FlowDirection] = FlowDirection.FORWARD):
        changed_ports = []
        for port in ports:
            if port in self.__flow_port_connections[direction]:
                changed_ports.append(self._propagate_for_flow_port(port, direction))
            elif port in self.__condition_port_connections:
                changed_ports.append(self._propagate_for_condition_port(port, direction))
            else:
                raise Exception(f'[ERROR] Found unconnected port: {port}')
        return list(set(list(chain.from_iterable(changed_ports))))

    def _propagate_for_flow_port(self, port: FlowPort, direction: FlowDirection):
        next_port = self.__flow_port_connections[direction][port]
        next_port.set_data(port.flush())
        return self._trigger_transform_call(next_port, direction)

    def _propagate_for_condition_port(self, port: FlowPort, direction: FlowDirection):
        sink_ports = self.__condition_port_connections[port]
        data = port.flush()
        changed_ports =[]
        for next_port in sink_ports:
            next_port.set_data(data)
            changed_ports.append(self._trigger_transform_call(next_port, direction))
        return list(chain.from_iterable(changed_ports))

    def _trigger_transform_call(self, port: FlowPort, direction: FlowDirection):
        interface = self.__transform_interfaces_by_port[port]
        changed_ports = []
        if (interface is not self.__interface) and interface.all_inputs_ready(direction):
            changed_ports = interface.call_transform(direction)
        return changed_ports

    def forward_transform(self, *data: Any) -> List[Any]:
        return self.forward(*data, direction=FlowDirection.FORWARD)

    def reverse_transform(self, *data: Any) -> List[Any]:
        return self.forward(*data, direction=FlowDirection.REVERSE)

    def add_determinant_tracker(self, ll_per_dim: LogLikelihoodPerDimension):
        super(NormalizingFlow, self).add_determinant_tracker(ll_per_dim)
        for t in self.__transform_modules:
            t.add_determinant_tracker(ll_per_dim)

    def clear_determinant_trackers(self):
        super(NormalizingFlow, self).clear_determinant_trackers()
        for t in self.__transform_modules:
            t.clear_determinant_trackers()

    def get_flow_port_by_label(self, label: str) -> FlowPort:
        return self.__flow_ports_by_label[label]

    def get_condition_port_by_label(self, label: str) -> FlowPort:
        return self.__condition_ports_by_label[label]

    def get_transform_by_label(self, label: str) -> InvertibleTransform:
        return self.__transforms_by_label[label]

    @classmethod
    def chain(
            cls,
            *transforms: InvertibleTransform,
            direction: Optional[FlowDirection] = FlowDirection.FORWARD,
            port_labels: Optional[Dict[FlowDirection, List[str]]] = None,
    ) -> 'NormalizingFlow':
        flow = cls()
        chainer = TransformChainer(flow)
        chainer.chain_transforms(list(transforms), port_labels, direction)
        return flow


class TransformChainer(object):

    def __init__(self, flow: NormalizingFlow):
        self.flow = flow

    @staticmethod
    def _parse_list_inputs(inputs: Union[List[Any], None], name: str, num_required: int) -> List[Any]:
        if inputs is not None:
            assert type(inputs) == list, \
                f'[ERROR] Input {name} must be given as a list.'
            assert len(inputs) == num_required, \
                f'[ERROR] Number of input {name} ({len(inputs)}) must match the number of required {name} ({num_required}).'
        else:
            inputs = [None] * num_required
        return inputs

    def _parse_dict_inputs(
            self,
            input_dict: Union[Dict[FlowDirection, str], None],
            name: str,
            num_required: Dict[FlowDirection, int]
    ):
        if input_dict is None:
            input_dict = {}
        for direction in [FlowDirection.FORWARD, FlowDirection.REVERSE]:
            ids = input_dict[direction] if direction in input_dict else None
            input_dict.update({direction: self._parse_list_inputs(ids, name, num_required[direction])})
        return input_dict

    def chain_transforms(
            self,
            transforms: List['InvertibleTransform'],
            port_labels: Union[Dict[FlowDirection, List[str]], None],
            direction: FlowDirection,
    ):
        num_in_ports = transforms[0].get_number_of_flow_ports(direction=direction)
        if port_labels is None:
            port_labels = {}
        if direction in port_labels:
            in_port_labels = port_labels[direction]
            num_port_labels = len(in_port_labels)
            assert num_port_labels == num_in_ports, \
                f'[ERROR] Number of port labels in direction {direction} ({num_port_labels})' \
                f' must match the number of required input ports ({num_in_ports})'
            ports = [self.flow.add_flow_port(direction, label=label) for label in in_port_labels]
        else:
            ports = None
        for transform in transforms:
            num_conditions = transform.get_number_of_condition_ports()
            if num_conditions > 0:
                condition_ports = [self.flow.add_condition_port() for _ in range(num_conditions)]
            else:
                condition_ports = None
            ports = self.flow.add_transform(
                transform, inputs=ports, direction=direction, conditions=condition_ports,
            )
        if direction.opposite() in port_labels:
            out_port_labels = port_labels[direction.opposite()]
            num_port_labels = len(out_port_labels)
            assert num_port_labels == len(ports), \
                f'[ERROR] Number of port labels in direction {direction} ({num_port_labels})' \
                f' must match the number of required output ports ({len(ports)})'
            out_ports = [self.flow.add_flow_port(direction.opposite(), label=label) for label in out_port_labels]
            for port, out_port in zip(ports, out_ports):
                self.flow.connect_flow_ports(port, out_port, direction=direction)
        return None