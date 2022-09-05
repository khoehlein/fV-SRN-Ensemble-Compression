from typing import Tuple

import torch
from torch.nn import functional as F
from torch import Tensor

from ..interface import InvertibleTransform, FlowDirection


class _SigmoidOrLogit(InvertibleTransform):

    def __init__(self):
        super(_SigmoidOrLogit, self).__init__({
            FlowDirection.FORWARD: 1,
            FlowDirection.REVERSE: 1,
        }, num_condition_ports=0)

    def _sigmoid_transform(self, x):
        out = (torch.sigmoid(x),)
        return out

    def _sigmoid_determinant(self, x: Tensor):
        log_dout_dx = F.logsigmoid(x) + F.logsigmoid(-x)
        return log_dout_dx

    def _logit_transform(self, x):
        out = (torch.logit(x),)
        return out

    def _logit_determinant(self, x: Tensor):
        if torch.any(torch.logical_or(0. >= x, x >= 1.)):
            raise Exception()
        log_dout_dx = - torch.log(x) - torch.log1p(- x)
        return log_dout_dx

    def _update_determinant_trackers(self, x):
        value = torch.mean(self._log_determinant(x), dim=list(range(1, len(x.shape))))
        dimensions = self.determinant_trackers[0].compute_dimensions(x)
        self.update_determinant_trackers(value, dimensions)

    def _log_determinant(self, x: Tensor):
        raise NotImplementedError


class Sigmoid(_SigmoidOrLogit):

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = self._sigmoid_transform(data[0])
        if self.has_determinant_tracker():
            self._update_determinant_trackers(data[0])
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = self._logit_transform(data[0])
        if self.has_determinant_tracker():
            self._update_determinant_trackers(out[0])
        return out

    def _log_determinant(self, x):
        return self._sigmoid_determinant(x)


class Logit(_SigmoidOrLogit):

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = self._logit_transform(data[0])
        if self.has_determinant_tracker():
            self._update_determinant_trackers(data[0])
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        out = self._sigmoid_transform(data[0])
        if self.has_determinant_tracker():
            self._update_determinant_trackers(out[0])
        return out

    def _log_determinant(self, x):
        return self._logit_determinant(x)
