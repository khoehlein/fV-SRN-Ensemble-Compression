import math
from typing import List

import torch
from torch import nn, Tensor

from .custom_activations import Sine
from .processor_sequential_wrapper import ProcessorSequentialWrapper


class ResidualSineLayer(nn.Module):
    """
    From Lu & Berger 2021, Compressive Neural Representations of Volumetric Scalar Fields
    https://github.com/matthewberger/neurcomp/blob/main/siren.py
    """
    def __init__(self, num_channels: int, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.num_channels = num_channels
        self.linear_1 = nn.Linear(num_channels, num_channels, bias=bias)
        self.linear_2 = nn.Linear(num_channels, num_channels, bias=bias)

        self.weight_1 = .5 if ave_first else 1.
        self.weight_2 = .5 if ave_second else 1.

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            b = math.sqrt(6 / self.num_channels) / self.omega_0
            self.linear_1.weight.uniform_(-b, b)
            self.linear_2.weight.uniform_(-b, b)

    def forward(self, input: Tensor):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)


class ResidualSineProcessor(ProcessorSequentialWrapper):
    """
    copied and modified from https://github.com/matthewberger/neurcomp/blob/main/siren.py
    """
    def __init__(
            self,
            data_input_channels: int, latent_input_channels: int, output_channels: int,
            layer_sizes: List[int]
    ):
        if len(set(layer_sizes)) != 1:
            raise ValueError("[ERROR] For ResidualSine, all layers must have the same size")

        layers = []
        current_channels = data_input_channels + latent_input_channels
        for i, new_channels in enumerate(layer_sizes):
            if i == 0:
                current_layer = nn.Linear(current_channels, new_channels)
                with torch.no_grad():
                    current_layer.weight.uniform_(-1. / current_channels, 1. / current_channels)
                layers.append((f'linear{i}', current_layer))
                layers.append((f'Sine{i}', Sine(omega=30)))
            else:
                ave_first = (i > 1)
                ave_second = (i == (len(layer_sizes) - 2))
                layers.append((f'ResidualSine{i}', ResidualSineLayer(new_channels, bias=True, ave_first=ave_first, ave_second=ave_second),))
            current_channels = new_channels
        last_layer = nn.Linear(current_channels, output_channels)
        with torch.no_grad():
            b = math.sqrt(6 / current_channels) / 30.0
            last_layer.weight.uniform_(-b, b)
        layers.append((f'linear{len(layer_sizes)}', last_layer))

        super(ResidualSineProcessor, self).__init__(
            data_input_channels, latent_input_channels, output_channels,
            layers, layer_sizes, 'ResidualSine', []
        )
