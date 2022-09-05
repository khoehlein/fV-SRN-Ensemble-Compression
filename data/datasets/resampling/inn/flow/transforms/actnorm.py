from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn as nn
from .clamping import ArctanClamping
from ..interface import InvertibleTransform, FlowDirection


class ActNorm2d(InvertibleTransform):

    def __init__(self, channels: int, eps: Optional[float]=1.e-6, clamp: Optional[Union[float, None]]=2., label: Optional[str] = None):
        super(ActNorm2d, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        self.register_parameter('offset', nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True))
        self.register_parameter('raw_log_scale', nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True))
        self.clamping = ArctanClamping(clamp, log_scale=True) if clamp is not None else None
        self.eps = eps
        self._invert_transform = None

    def _initialize_parameters(self, x: Tensor, invert_transform: bool) -> None:
        sample_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
        sample_log_std = torch.log(torch.std(x, dim=(0, 2, 3), keepdim=True, unbiased=True) + self.eps)
        self.offset.data = sample_mean.data
        self.raw_log_scale.data = sample_log_std.data
        self._invert_transform = invert_transform

    def forward_transform(self, x: Tensor) -> Tuple[Tensor]:
        if self._invert_transform is None:
            self._initialize_parameters(x, False)
        rescaling = self._reverse_rescaling if self._invert_transform else self._forward_rescaling
        out = (rescaling(x),)
        if self.has_determinant_tracker():
            self._update_log_determinant(x)
        return out

    def reverse_transform(self, x: Tensor) -> Tuple[Tensor]:
        if self._invert_transform is None:
            self._initialize_parameters(x, True)
        rescaling = self._forward_rescaling if self._invert_transform else self._reverse_rescaling
        out = (rescaling(x),)
        if self.has_determinant_tracker():
            self._update_log_determinant(x)
        return out

    def _forward_rescaling(self, x: Tensor) -> Tensor:
        return (x - self.offset) * torch.exp(-self._log_scale())

    def _reverse_rescaling(self, x: Tensor) -> Tensor:
        return torch.exp(self._log_scale()) * x + self.offset

    def _log_scale(self) -> Tensor:
        if self.clamping is not None:
            return self.clamping(self.raw_log_scale)
        return self.raw_log_scale

    def _log_determinant(self) -> Tensor:
        raw_log_det = torch.mean(self._log_scale())
        return raw_log_det if self._invert_transform else - raw_log_det

    def _update_log_determinant(self, x: Tensor) -> None:
        value = torch.tensor([self._log_determinant()] * x.shape[0], device=x.device, dtype=x.dtype)
        dimensions = self.determinant_trackers[0].compute_dimensions(x, batch_dim=0)
        self.update_determinant_trackers(value, dimensions)
