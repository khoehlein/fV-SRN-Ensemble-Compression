from typing import Any, Tuple

import torch
from torch import nn

from ..interface import InvertibleTransform, FlowDirection
from .clamping import ArctanClamping


class PReLU(InvertibleTransform):

    def __init__(self, channels, clamp=2., log_scale=True):
        super(PReLU, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, num_condition_ports=0)
        self.register_parameter('raw_scale', nn.Parameter(torch.randn(channels) * 1.e-6, requires_grad=True))
        self.clamping = ArctanClamping(clamp, log_scale=log_scale)

    def forward_transform(self, *data: Any) -> Tuple[Any, ...]:
        log_scale = self._log_scale(data[0])
        out = (data[0] * torch.exp(log_scale),)
        if self.has_determinant_tracker():
            self._update_log_determinant(log_scale)
        return out

    def reverse_transform(self, *data: Any) -> Tuple[Any, ...]:
        log_scale = self._log_scale(data[0])
        out = (data[0] * torch.exp(- log_scale),)
        if self.has_determinant_tracker():
            self._update_log_determinant(log_scale)
        return out

    def _update_log_determinant(self, log_scale):
        value = torch.mean(log_scale, dim=list(range(1, 4)))
        dimensions = self.determinant_trackers[0].compute_dimensions(log_scale, batch_dim=0)
        self.update_determinant_trackers(value, dimensions)

    def _log_scale(self, data):
        return torch.where(data >= 0, torch.zeros_like(data), self.clamping(self.raw_scale)[None, :, None, None])

