from typing import Optional, Tuple, List
import torch
from torch import Tensor
from torch import nn as nn
from .clamping import ArctanClamping
from ..interface import InvertibleTransform, FlowDirection


class CouplingBlock(InvertibleTransform):

    __version__ = '1.0'

    def __init__(self, channels: int, processing_module: nn.Module, swap_groups: Optional[bool]=False, clamp: Optional[float]=2., label: Optional[str] = None):
        super(CouplingBlock, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        assert channels % 2 == 0
        self.in_channels = channels
        self.out_channels = channels
        self.processing_module = processing_module
        self.swap_groups = swap_groups
        self.clamping = ArctanClamping(clamp, log_scale=True)

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor]:
        x1, x2 = self._split_input(data[0])
        if self.swap_groups:
            x1, x2 = x2, x1
        log_scale, translation = self._compute_scales(x1)
        x2 = torch.exp(log_scale) * (x2 + translation)
        if self.swap_groups:
            x1, x2 = x2, x1
        out = (self._combine_outputs(x1, x2),)
        if self.has_determinant_tracker():
            self._update_log_determinant(log_scale)
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor]:
        x1, x2 = self._split_input(data[0])
        if self.swap_groups:
            x1, x2 = x2, x1
        log_scale, translation = self._compute_scales(x1)
        x2 = x2 * torch.exp(- log_scale) - translation
        if self.swap_groups:
            x1, x2 = x2, x1
        out = (self._combine_outputs(x1, x2),)
        if self.has_determinant_tracker():
            self._update_log_determinant(log_scale)
        return out

    def _compute_scales(self, *data: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.processing_module(data[0])
        raw_scale, translation = torch.chunk(features, 2, dim=1)
        log_scale = self.clamping(raw_scale)
        return log_scale, translation

    @staticmethod
    def _split_input(x: Tensor) -> List[Tensor]:
        return torch.chunk(x, 2, dim=1)

    @staticmethod
    def _combine_outputs(x1, x2):
        return torch.cat((x1, x2), dim=1)

    @staticmethod
    def _log_determinant(log_scale: Tensor) -> Tensor:
        shape = log_scale.shape
        return torch.mean(log_scale, dim=tuple(range(1, len(shape))))

    def _update_log_determinant(self, log_scale: Tensor) -> None:
        value = self._log_determinant(log_scale)
        dimensions = self.determinant_trackers[0].compute_dimensions(log_scale, batch_dim=0)
        # 'value' is the log det PER DIMENSION for ONLY the transformed half of variables.
        # The weighting by 'dimensions' of only the same variables compensates for the dimension mismatch
        # between transformed variables for mean computation and full variable dimensionality.
        self.update_determinant_trackers(value, dimensions)


class ClimAlignCouplingBlock(CouplingBlock):

    def __init__(
            self,
            channels: int,
            processing_module: nn.Module,
            label: Optional[str] = None
    ):
        super(ClimAlignCouplingBlock, self).__init__(channels, processing_module, swap_groups=False, clamp=1., label=label)
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, channels // 2, 1, 1)))
        self.sigmoid = nn.Sigmoid()

    def _compute_scales(self, *data: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.processing_module(data[0])
        raw_scale, translation = torch.chunk(features, 2, dim=1)
        # Scale activation as found in:
        # https://github.com/bgroenks96/normalizing-flows/blob/8fc5795687bda4cce1c236147fa8fbc47a5efba9/normalizing_flows/flows/glow/affine_coupling.py
        log_scale = torch.log(self.sigmoid(torch.exp(self.log_scale) * raw_scale) + 0.1)
        return log_scale, translation


class ConditionalCouplingBlock(CouplingBlock):

    def __init__(
            self, channels: int,
            processing_module: nn.Module,
            swap_groups: Optional[bool]=False,
            clamp: Optional[float]=2.,
            label: Optional[str] = None,
    ):
        super(ConditionalCouplingBlock, self).__init__(
            channels, processing_module,
            swap_groups=swap_groups, clamp=clamp,
            label=label
        )
        self.set_num_condition_ports(1)

    def _compute_scales(self, *data: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = torch.cat(data, dim=1)
        features = self.processing_module(inputs)
        raw_scale, translation = torch.chunk(features, 2, dim=1)
        log_scale = self.clamping(raw_scale)
        return log_scale, translation

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor]:
        x1, x2 = self._split_input(data[0])
        if self.swap_groups:
            x1, x2 = x2, x1
        log_scale, translation = self._compute_scales(x1, data[1])
        x2 = torch.exp(log_scale) * x2 + translation
        if self.swap_groups:
            x1, x2 = x2, x1
        out = (self._combine_outputs(x1, x2),)
        if self.has_determinant_tracker():
            self._update_log_determinant(log_scale)
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor]:
        x1, x2 = self._split_input(data[0])
        if self.swap_groups:
            x1, x2 = x2, x1
        log_scale, translation = self._compute_scales(x1, data[1])
        x2 = (x2 - translation) * torch.exp(-log_scale)
        if self.swap_groups:
            x1, x2 = x2, x1
        out = (self._combine_outputs(x1, x2),)
        if self.has_determinant_tracker():
            self._update_log_determinant(log_scale)
        return out


class ConditionalClimAlignCouplingBlock(ConditionalCouplingBlock):

    def __init__(
            self,
            channels: int,
            processing_module: nn.Module,
            label: Optional[str] = None
    ):
        super(ConditionalClimAlignCouplingBlock, self).__init__(channels, processing_module, swap_groups=False, clamp=1., label=label)
        self.register_parameter('log_scale', nn.Parameter(torch.zeros(1, channels // 2, 1, 1)))
        self.sigmoid = nn.Sigmoid()

    def _compute_scales(self, *data: Tensor) -> Tuple[Tensor, Tensor]:
        inputs = torch.cat(data, dim=1)
        features = self.processing_module(inputs)
        raw_scale, translation = torch.chunk(features, 2, dim=1)
        # Scale activation as found in:
        # https://github.com/bgroenks96/normalizing-flows/blob/8fc5795687bda4cce1c236147fa8fbc47a5efba9/normalizing_flows/flows/glow/affine_coupling.py
        log_scale = torch.log(self.sigmoid(torch.exp(self.log_scale) * raw_scale) + 0.1)
        return log_scale, translation
