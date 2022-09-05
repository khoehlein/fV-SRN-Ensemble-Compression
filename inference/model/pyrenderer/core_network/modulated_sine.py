from typing import List

import torch
from torch import Tensor, nn

from .processor_sequential_wrapper import ProcessorSequentialWrapper


class ModulatedSine(nn.Module):

    def __init__(
            self,
            input_channels: int, output_channels: int, latent_channels: int,
            is_first: bool, omega=1.
    ):
        super().__init__()
        self._omega = float(omega)
        self._latent_size = latent_channels
        self._is_first = is_first
        self._relu = torch.nn.ReLU()
        if is_first:
            self._lin1 = torch.nn.Linear(
                input_channels - latent_channels, output_channels, True) # synthesizer
            self._lin2 = torch.nn.Linear(
                latent_channels, output_channels, True) # modulator
            self._isize = input_channels - latent_channels
        else:
            self._lin1 = torch.nn.Linear(
                input_channels, output_channels, True)  # synthesizer
            self._lin2 = torch.nn.Linear(
                input_channels + latent_channels, output_channels, True)  # modulator
            self._isize = input_channels

    def _decompose_input(self, x: Tensor):
        if self._is_first:
            sine_input = x[:, :self._isize]
            latent_features = x[:, self._isize:]
            modulation = latent_features
        else:
            sine_input = x[:, :self._isize]
            modulation = x[:, self._isize:]
            latent_features = x[:, -self._latent_size:]
        return sine_input, modulation, latent_features

    def forward(self, x):
        sine_input, modulation_input, latent_features = self._decompose_input(x)
        modulation = self._relu(self._lin2(modulation_input))
        modulated_output = modulation * torch.sin(self._lin1(sine_input))
        res = torch.cat((modulated_output, modulation, latent_features), dim=1)
        return res


class Select(nn.Module):

    def __init__(self, from_idx: int, to_idx: int):
        super().__init__()
        self._from = from_idx
        self._to = to_idx

    def forward(self, x):
        return x[:, self._from:self._to]


class ModulatedSineProcessor(ProcessorSequentialWrapper):

    def __init__(
            self,
            data_input_channels: int, latent_input_channels: int, output_channels: int,
            layer_sizes: List[int]
    ):
        layers = []
        current_channels = data_input_channels
        for i, s in enumerate(layer_sizes):
            s = s // 2  # because modulated size doubles it for synthesizer+modulator
            # this way, the methods are comparable
            layers.append((f'linear{i}', ModulatedSine(
                current_channels, s, latent_input_channels, i == 0)))
            current_channels = s
        layers.append(('select_synthesizer', Select(0, current_channels)))
        last_layer = nn.Linear(current_channels, output_channels)
        layers.append((f'linear{len(layer_sizes)}', last_layer))
        super(ModulatedSineProcessor, self).__init__(
            data_input_channels, latent_input_channels, output_channels,
            layers, layer_sizes, 'ModulatedSine', []
        )