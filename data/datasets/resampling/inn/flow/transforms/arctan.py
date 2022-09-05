from typing import Tuple

import torch
from numpy import pi as PI
from torch import Tensor

from ..interface import InvertibleTransform, FlowDirection


class _TanArcTanRescaling(InvertibleTransform):

    def __init__(self):
        super(_TanArcTanRescaling, self).__init__({
            FlowDirection.FORWARD: 1,
            FlowDirection.REVERSE: 1,
        }, num_condition_ports=0)

    def _tan_transform(self, x):
        return torch.tan(PI * (x - 0.5)) / PI

    def _tan_log_derivative(self, x):
        return - 2. * torch.log(torch.cos(PI * (x - 0.5)))

    def _arctan_transform(self, x):
        return torch.arctan(PI * x) / PI + 0.5

    def _arctan_log_derivative(self, x):
        return - torch.log1p((PI * x) ** 2)

    def _update_determinant_trackers(self, x):
        value = torch.mean(self._log_derivative(x), dim=list(range(1, len(x.shape))))
        dimensions = self.determinant_trackers[0].compute_dimensions(x)
        self.update_determinant_trackers(value, dimensions)

    def _log_derivative(self, x: Tensor):
        raise NotImplementedError


class Tan(_TanArcTanRescaling):

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = (self._tan_transform(data[0]),)
        if self.has_determinant_tracker():
            self._update_determinant_trackers(data[0])
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = (self._arctan_transform(data[0]),)
        if self.has_determinant_tracker():
            self._update_determinant_trackers(out[0])
        return out

    def _log_derivative(self, x):
        return self._tan_log_derivative(x)


class ArcTan(_TanArcTanRescaling):

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = (self._arctan_transform(data[0]),)
        if self.has_determinant_tracker():
            self._update_determinant_trackers(data[0])
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = (self._tan_transform(data[0]),)
        if self.has_determinant_tracker():
            self._update_determinant_trackers(out[0])
        return out

    def _log_derivative(self, x):
        return self._arctan_log_derivative(x)
