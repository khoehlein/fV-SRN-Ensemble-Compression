from typing import List, Any

from torch import nn

from .processor_sequential_wrapper import ProcessorSequentialWrapper
from . import custom_activations as ca


class SimpleMLP(ProcessorSequentialWrapper):

    def __init__(
            self,
            data_input_channels: int, latent_input_channels: int, output_channels: int,
            layer_sizes: List[int], activation: str, activation_params: List[Any]
    ):
        activ_class = getattr(nn, activation, None)
        if activ_class is None:
            activ_class = getattr(ca, activation, None)
        assert activ_class is not None, \
            f'[ERROR] Activation <{activation}> has neither been found in torch.nn nor in custom activations.'
        layers = []
        current_channels = data_input_channels + latent_input_channels
        for i, s in enumerate(layer_sizes):
            layers.append((f'linear{i}', nn.Linear(current_channels, s)))
            layers.append((f'{activation.lower()}{i}', activ_class(*activation_params)))
            current_channels = s
        last_layer = nn.Linear(current_channels, output_channels)
        layers.append((f'linear{len(layer_sizes)}', last_layer))

        super(SimpleMLP, self).__init__(
            data_input_channels, latent_input_channels, output_channels,
            layers, layer_sizes, activation, activation_params
        )