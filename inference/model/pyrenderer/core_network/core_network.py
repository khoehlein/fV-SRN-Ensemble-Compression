import argparse
from typing import Union, Optional, Dict, Any, List

import pyrenderer
import torch
from torch import Tensor, nn

from data.datasets import OutputMode
from .modulated_sine import ModulatedSineProcessor
from .residual_sine import ResidualSineProcessor
from .simple_mlp import SimpleMLP
from inference.model.input_parameterization import IInputParameterization
from inference.model.latent_features import ILatentFeatures
from ...core_network import ICoreNetwork
from ...output_parameterization import IOutputParameterization


class PyrendererCoreNetwork(ICoreNetwork):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CoreNetwork')
        prefix = '--network:core:'
        group.add_argument(prefix + 'layer-sizes', default='32:32:32', type=str,
                           help="The size of the hidden layers, separated by colons ':'")
        group.add_argument(prefix + 'activation', default="ReLU", type=str, help="""
                                The activation function for the hidden layers.
                                This is the class name for activations in torch.nn.** .
                                The activation for the last layer is fixed by the output mode.
                                To pass extra arguments, separate them by colons, e.g. 'Snake:2'""")
        group.add_argument(prefix + 'split-members', action='store_true', dest='network_core_split_members')
        group.set_defaults(network_core_split_members=False)

    @staticmethod
    def _build_processor(
            args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
    ):
        prefix = 'network:core:'
        layer_sizes = list(map(int, args[prefix + 'layer_sizes'].split(':')))
        activation, *activation_params = args[prefix + 'activation'].split(':')
        activation_params = [float(p) for p in activation_params]
        data_input_channels = input_parameterization.output_channels()
        latent_input_channels = latent_features.output_channels() if latent_features is not None else 0
        output_channels = output_parameterization.output_channels()
        if activation == "ModulatedSine":
            processor = ModulatedSineProcessor(
                data_input_channels, latent_input_channels, output_channels,
                layer_sizes,
            )
        elif activation == "ResidualSine":
            processor = ResidualSineProcessor(
                data_input_channels, latent_input_channels, output_channels,
                layer_sizes
            )
        else:
            processor = SimpleMLP(
                data_input_channels, latent_input_channels, output_channels,
                layer_sizes, activation, activation_params
            )
        if output_parameterization.output_mode() == OutputMode.RGBO: #rgba
            last_layer = processor.last_layer()
            last_layer.bias.sample_summary = torch.abs(last_layer.bias.sample_summary) + 1.0 # positive output to see something
        return processor

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
            member_keys=None
    ):
        if args['network_core_split_members']:
            assert member_keys is not None
            return PyrendererMultiCoreNetwork.from_dict(args, input_parameterization, latent_features, output_parameterization, member_keys)
        else:
            return PyrendererSingleCoreNetwork.from_dict(args, input_parameterization, latent_features, output_parameterization)


class PyrendererSingleCoreNetwork(PyrendererCoreNetwork):

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
            member_keys=None,
    ):
        processor = cls._build_processor(args, input_parameterization, latent_features, output_parameterization)
        return cls(processor)

    def __init__(self, processor: ICoreNetwork):
        super(PyrendererSingleCoreNetwork, self).__init__()
        self.processor = processor

    def forward(
            self, data_input: Tensor, latent_input: Union[Tensor, None],
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor
    ) -> Tensor:
        return self.processor(data_input, latent_input, positions, transfer_functions, time, member)

    def data_input_channels(self) -> int:
        return self.processor.data_input_channels()

    def latent_input_channels(self) -> int:
        return self.processor.latent_input_channels()

    def output_channels(self) -> int:
        return self.processor.output_channels()

    def last_layer(self):
        return self.processor.last_layer()

    def export_to_pyrenderer(self, network: Optional[pyrenderer.SceneNetwork] = None,
                             time=None, ensemble=None, pad_first_layer:int=0) -> pyrenderer.SceneNetwork:
        # ensemble not encoded in the network -> ignore
        return self.processor.export_to_pyrenderer(
            network=network, time=time, pad_first_layer=pad_first_layer)


class PyrendererMultiCoreNetwork(PyrendererCoreNetwork):

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
            member_keys=None
    ):
        processors = [
            cls._build_processor(args, input_parameterization, latent_features, output_parameterization)
            for _ in member_keys
        ]
        return cls(processors, member_keys)

    def __init__(self, processors: List[ICoreNetwork], member_keys: List[Any]):
        super(PyrendererMultiCoreNetwork, self).__init__()
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.processors = nn.ModuleList(processors)
        print(self.processors)

    def forward(
            self, data_input: Tensor, latent_input: Union[Tensor, None],
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor
    ) -> Tensor:
        unique_members = torch.unique(member)
        if len(unique_members) == 1:
            processor = self.processors[int(unique_members[0].item())]
            return processor.forward(data_input, latent_input, positions, transfer_functions, time, member)
        out = torch.empty(len(positions), self.output_channels(), device=positions.device, dtype=positions.dtype)
        for umem in unique_members:
            processor = self.processors[int(umem.item())]
            locations = torch.eq(umem, member)
            out[locations] = processor.forward(
                data_input[locations], latent_input[locations],
                positions[locations], transfer_functions[locations], time[locations], member[locations]
            )
        return out

    def data_input_channels(self) -> int:
        return self.processors[0].data_input_channels()

    def latent_input_channels(self) -> int:
        return self.processors[0].latent_input_channels()

    def output_channels(self) -> int:
        return self.processors[0].output_channels()

    def last_layer(self):
        raise NotImplementedError()

    def export_to_pyrenderer(self, network: Optional[pyrenderer.SceneNetwork] = None,
                             time=None, ensemble=None, pad_first_layer:int=0) -> pyrenderer.SceneNetwork:
        if ensemble is None and len(self.processors)>1:
            raise ValueError("no ensemble specified, but multiple processors are defined")
        processor = self.processors[ensemble or 0]
        return processor.export_to_pyrenderer(network=network, time=time, pad_first_layer=pad_first_layer)


    def num_members(self) -> int:
        return len(self.processors)

    def uses_member(self) -> bool:
        return len(self.processors)>1