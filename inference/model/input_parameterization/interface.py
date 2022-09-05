from torch import nn, Tensor


class IComponentProcessor(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(IComponentProcessor, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

    def in_channels(self):
        return self._in_channels

    def out_channels(self):
        return self._out_channels


class IInputParameterization(nn.Module):

    def forward(self, positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor):
        raise NotImplementedError()

    def output_channels(self) -> int:
        raise NotImplementedError()

    def uses_positions(self):
        raise NotImplementedError()

    def uses_transfer_functions(self):
        raise NotImplementedError()

    def uses_time(self) -> bool:
        raise NotImplementedError()

    def uses_member(self):
        raise NotImplementedError()

    def uses_directions(self) -> bool:
        raise NotImplementedError()
