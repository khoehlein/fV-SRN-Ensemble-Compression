from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn as nn
import torch.linalg as la
from .clamping import ArctanClamping
from ..interface import InvertibleTransform, FlowDirection


class FullConv2d(InvertibleTransform):

    __version__ = '1.0'

    def __init__(self, channels: int, use_bias: Optional[bool]=True, label: Optional[str] = None):
        super(FullConv2d, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        assert channels > 0 and (int(channels) == channels)
        channels = int(channels)
        self.in_channels = channels
        self.out_channels = channels
        self.register_parameter(
            'weight',
            nn.Parameter(self._get_initial_matrix(), requires_grad=True)
        )
        self.register_parameter(
            'bias',
            nn.Parameter(torch.zeros(channels), requires_grad=True) if use_bias else None
        )

    def _get_initial_matrix(self) -> Tensor:
        channels = self.in_channels
        w = torch.randn(channels, channels)
        q, _ = la.qr(w, mode='complete')
        return q

    def forward_transform(self, x: Tensor) -> Tuple[Tensor]:
        out = (torch.conv2d(x, self.weight.view(self.out_channels, self.in_channels, 1, 1), bias=self.bias),)
        if self.has_determinant_tracker():
            self._update_log_determinant(x)
        return out

    def reverse_transform(self, x: Tensor) -> Tuple[Tensor]:
        shape = x.shape
        out = x
        if self.bias is not None:
            out = out - self.bias.view(1, self.out_channels, 1, 1)
        lu_data = torch.lu(torch.tile(self.weight[None, ...], (shape[0], 1, 1)))
        out = (torch.lu_solve(out.view(*shape[:2], -1), *lu_data).view(*shape),)
        if self.has_determinant_tracker():
            self._update_log_determinant(x)
        return out

    def _log_determinant(self) -> Tensor:
        return la.slogdet(self.weight).logabsdet / self.in_channels

    def _update_log_determinant(self, x: Tensor) -> Tensor:
        value =  torch.tensor([self._log_determinant()] * x.shape[0], dtype=x.dtype, device=x.device)
        dimensions = self.determinant_trackers[0].compute_dimensions(x, batch_dim=0)
        self.update_determinant_trackers(value, dimensions)


class LUConv2d(InvertibleTransform):

    __version__ = '1.0'

    def __init__(self, channels: int, use_bias: Optional[bool]=True, clamp: Optional[float]=2., label: Optional[str] = None):
        super(LUConv2d, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        assert channels > 0 and (int(channels) == channels)
        channels = int(channels)
        self.in_channels = channels
        self.out_channels = channels
        self.register_parameter(
            'weight',
            nn.Parameter(self._get_initial_matrix(), requires_grad=True)
        )
        self.register_parameter(
            'bias',
            nn.Parameter(torch.zeros(channels), requires_grad=True) if use_bias else None
        )
        nn.init.normal_(self.weight, 0., 1.e-4)
        self.clamping = ArctanClamping(clamp, log_scale=True)

    def _get_initial_matrix(self) -> Tensor:
        channels = self.in_channels
        w = torch.randn((channels, channels))
        q, _ = la.qr(w)
        lu, _ = torch.lu(q)
        idx = list(range(channels))
        lu[idx, idx] = torch.log(torch.abs(lu[idx, idx]))
        return lu

    def _log_diagonal(self):
        d = torch.diag(self.weight, diagonal=0)
        return self.clamping(d)

    def _get_l_r_matrizes(self):
        log_diag = self._log_diagonal()
        l = torch.diag(torch.ones_like(log_diag)) + torch.tril(self.weight, diagonal=-1)
        r = torch.diag(torch.exp(log_diag)) + torch.triu(self.weight, diagonal=1)
        return l, r

    def forward_transform(self, x: Tensor) -> Tuple[Tensor]:
        l, r = self._get_l_r_matrizes()
        w = torch.mm(l, r).view(self.out_channels, self.in_channels, 1, 1)
        b = self.bias if self.use_bias else None
        out = (torch.conv2d(x, w, bias=b),)
        if self.has_determinant_tracker():
            self._update_log_determinant(x)
        return out

    def reverse_transform(self, x: Tensor) -> Tuple[Tensor]:
        shape = x.shape
        out = torch.flatten(x, start_dim=2)
        if self.use_bias:
            out = out - self.bias.view(1, self.out_channels, 1)
        l, r = self._get_l_r_matrizes()
        out, _ = torch.triangular_solve(out, l, upper=False, unitriangular=True)
        out, _ = torch.triangular_solve(out, r, upper=True, unitriangular=False)
        out = (out.view(shape),)
        if self.has_determinant_tracker():
            self._update_log_determinant(x)
        return out

    def _log_determinant(self) -> Tensor:
        return torch.mean(self._log_diagonal())

    def _update_log_determinant(self, x: Tensor) -> None:
        value =  torch.tensor([self._log_determinant()] * x.shape[0], dtype=x.dtype, device=x.device)
        dimensions = self.determinant_trackers[0].compute_dimensions(x, batch_dim=0)
        self.determinant_tracker.add_flow_log_determinant(value, dimensions)
