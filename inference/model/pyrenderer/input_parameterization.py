import argparse
from typing import Optional, Any, Dict

import pyrenderer
import torch
from torch import Tensor

from inference.model.input_parameterization import IInputParameterization, RandomFourierFeatures, NerfFourierFeatures, \
    IFourierFeatures, DirectForward
from inference.model.input_parameterization.trainable_fourier_features import TrainableFourierFeatures


class PyrendererInputParameterization(IInputParameterization):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('InputParameterization')
        prefix = '--network:input:'
        group.add_argument(
            prefix + 'use-direct-time', dest='network_inputs_use_direct_time', action='store_true',
            help="""use time as direct input to the core processing module"""
        )
        group.set_defaults(network_inputs_use_direct_time=False)
        group.add_argument(
            prefix + 'fourier:positions:num-features', type=int, default=0,
            help="""number of Fourier features on position input"""
        )
        group.add_argument(
            prefix + 'fourier:positions:method', type=str, default=None, choices=['nerf', 'random'],
            help="""method for constructing the Fourier matrices for positions"""
        )
        group.add_argument(
            prefix + 'fourier:time:num-features', type=int, default=0,
            help="""number of Fourier features on time input"""
        )
        group.add_argument(
            prefix + 'fourier:time:method', type=str, default=None, choices=['nerf', 'random', 'parametric'],
            help="""method for constructing the Fourier matrices for time"""
        )
        group.add_argument(
            prefix + 'fourier:method', type=str, default='nerf', choices=['nerf', 'random', 'parametric'],
            help="""method for constructing the Fourier matrices"""
        )
        group.add_argument(
            prefix + 'fourier:random:std', type=float, default=0.01,
            help="""standard deviation for Fourier matrices in random mode"""
        )

    @classmethod
    def from_dict(cls, args: Dict[str, Any]):
        def get_arg(name):
            return args['network:input:' + name]

        use_direct_time = args['network_inputs_use_direct_time']

        def build_fourier_processor(name: str, in_channels: int):
            num_features = get_arg(f'fourier:{name}:num_features')
            if num_features <= 0:
                return None
            method = get_arg(f'fourier:{name}:method')
            if method is None:
                method = get_arg(f'fourier:method')
            if method == 'random':
                std = get_arg('fourier:random:std')
                return RandomFourierFeatures(in_channels, num_features, std=std)
            elif method == 'nerf':
                return NerfFourierFeatures(in_channels, num_features)
            elif method == 'parametric':
                return TrainableFourierFeatures(in_channels, num_features)

        fourier_positions = build_fourier_processor('positions', 3)
        fourier_time = build_fourier_processor('time', 1)
        return cls(
            fourier_positions=fourier_positions, fourier_time=fourier_time,
            use_direct_time=use_direct_time
        )

    def __init__(
            self,
            fourier_positions: Optional[IFourierFeatures] = None,
            fourier_time: Optional[IFourierFeatures] = None,
            use_direct_time: Optional[bool] = False
    ):
        super(PyrendererInputParameterization, self).__init__()
        self.direct_positions = DirectForward(3)
        self.direct_time = DirectForward(1) if use_direct_time else None
        self.fourier_positions = fourier_positions
        self.fourier_time = fourier_time

    def forward(self, positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor):
        outputs = [self.direct_positions(positions)]
        self._evaluate_if_not_none(self.direct_time, time[:, None], outputs)
        self._evaluate_if_not_none(self.fourier_positions, positions, outputs)
        self._evaluate_if_not_none(self.fourier_time, time[:, None], outputs)
        outputs = torch.cat(outputs, dim=-1)
        return outputs

    @staticmethod
    def _evaluate_if_not_none(m, x, out):
        if m is not None:
            out.append(m.forward(x))

    def output_channels(self) -> int:
        out_channels = 0
        out_channels += self._get_out_channels_if_not_none(self.direct_positions)
        out_channels += self._get_out_channels_if_not_none(self.direct_time)
        out_channels += self._get_out_channels_if_not_none(self.fourier_positions)
        out_channels += self._get_out_channels_if_not_none(self.fourier_time)
        return out_channels

    @staticmethod
    def _get_out_channels_if_not_none(m):
        return 0 if m is None else m.out_channels()

    def uses_premultiplied_fourier_features(self):
        """
        All Fourier features should use pre-multiplied Fourier-matrices!
        """
        return True

    def uses_direct_time(self):
        return self.direct_time is not None

    def uses_fourier_time(self):
        return self.fourier_time is not None

    def uses_fourier_positions(self):
        return self.fourier_positions is not None

    def uses_positions(self):
        return True

    def uses_time(self):
        return self.uses_direct_time() or self.uses_fourier_time()

    def uses_directions(self):
        """
        Support for directions is not implemented!
        """
        return False

    def uses_transfer_functions(self):
        """
        Support for transfer functions is not implemented!
        """
        return False

    def uses_member(self):
        """
        Support for member is not implemented!
        """
        return False

    def export_to_pyrenderer(self, network: Optional[pyrenderer.SceneNetwork] = None) -> pyrenderer.SceneNetwork:
        if network is None:
            network = pyrenderer.SceneNetwork()
        if self.fourier_positions is not None:
            B = self.fourier_positions.get_fourier_matrix()
            network.input.set_fourier_matrix_from_tensor(B, True)
        else:
            network.input.disable_fourier_features()
        network.input.has_time = self.uses_time()
        return network