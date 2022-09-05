from typing import Union

from torch import Tensor, nn


class IFieldEvaluator(nn.Module):
    """
    Baseclass for everything that accepts position samples in d_in-dimensionalspace
    and returns d_out-dimensional field output
    """

    def __init__(self, in_dimension: Union[int, None], out_dimension: Union[int, None], device):
        super(IFieldEvaluator, self).__init__()
        self._in_dimension = in_dimension
        self._out_dimension = out_dimension
        self.device = device

    def in_channels(self):
        return self._in_dimension

    def out_channels(self):
        return self._out_dimension

    def evaluate(self, positions: Tensor) -> Tensor:
        """
        Interface function for calling the field evaluation
        """
        positions = self._verify_positions(positions)
        out = self.forward(positions)
        return self._verify_outputs(out)

    def _verify_positions(self, positions: Tensor):
        assert len(positions.shape) == 2 and positions.shape[-1] == self.in_channels()
        if self.device is None or positions.device == self.device:
            return positions
        return positions.to(self.device)

    def _verify_outputs(self, outputs: Tensor) -> Tensor:
        if len(outputs.shape) < 2:
            outputs = outputs[:, None]
        assert len(outputs.shape) == 2 and outputs.shape[-1] == self.out_channels()
        return outputs

    def forward(self, positions: Tensor) -> Tensor:
        """
        Function to implement the field evaluation logic
        """
        raise NotImplementedError()
