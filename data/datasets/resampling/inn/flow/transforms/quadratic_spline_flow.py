from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from ..interface import InvertibleTransform, FlowDirection


class RationalQuadraticSplineFlow(InvertibleTransform):

    def __init__(self, channels, processor, swap_groups=False, num_condition_ports=0):
        super(RationalQuadraticSplineFlow, self).__init__({
            FlowDirection.FORWARD: 1, FlowDirection.REVERSE:1
        }, num_condition_ports=num_condition_ports)
        self.channels = channels
        self.processor = processor
        self.swap_groups = swap_groups

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        return self._apply_transform(data, self._apply_forward_transform)

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        return self._apply_transform(data, self._apply_reverse_transform)

    def _apply_transform(self, data: Tuple[Tensor, ...], transform) -> Tuple[Tensor, ...]:
        x1, x2 = self._split_data(data[0])
        if self.swap_groups:
            x1, x2 = x2, x1
        if len(data) > 1:
            assert len(data[1:]) == self.get_number_of_condition_ports()
            weight_inputs = torch.cat([x1, *data[1:]], dim=1)
        else:
            weight_inputs = x1
        v, w = self._compute_weights(weight_inputs)
        x2, q = transform(x2, v, w)
        if self.has_determinant_tracker():
            self._update_determinant_trackers(q)
        if self.swap_groups:
            x1, x2 = x2, x1
        out = (self._join_data(x1, x2),)
        return out

    def _split_data(self, data: Tensor):
        lower = self.channels // 2
        upper = self.channels - lower
        return torch.split(data, [lower, upper], dim=1)

    def _join_data(self, x1, x2):
        return torch.cat([x1, x2], dim=1)

    def _compute_weights(self, x1: Tensor):
        batch_size, _, *sizes = x1.shape
        features = self.processor(x1)
        features = features.view(batch_size, self._out_channels(), -1, *sizes)
        if len(features.shape) > 3:
            features = torch.stack(torch.unbind(features, dim=2), dim=-1)
        k = (features.shape[-1] - 1) // 2
        raw_v, raw_w = torch.split(features, [k + 1, k], dim=-1)
        w = torch.softmax(raw_w, dim=-1)
        raw_v = F.softplus(raw_v)
        v = raw_v / torch.sum((raw_v[..., :-1] + raw_v[..., 1:]) / 2. * w, dim=-1, keepdim=True)
        return v, w

    def _out_channels(self):
        if self.swap_groups:
            return self.channels // 2
        else:
            return self.channels - self.channels // 2

    def _apply_forward_transform(self, x2: Tensor, v: Tensor, w: Tensor):
        cum_w = torch.cumsum(w, dim=-1)
        cum_w[..., -1] = 1.
        b = torch.clip(torch.searchsorted(cum_w, torch.unsqueeze(x2, dim=-1).contiguous())[..., 0], max=(cum_w.shape[-1] - 1))
        v_ib, v_ibp1, w_ib = self._apply_indexing(v, w, b)
        dv = v_ibp1 - v_ib
        c0, c1, c2 = self._compute_coefficients(v, w, b, v_ib, w_ib, dv)
        alpha = (x2 - (self._select_from_tensor(cum_w, b) - w_ib)) / w_ib
        x2 = torch.pow(alpha, 2.) * c2 + alpha * c1 + c0
        q = v_ib + alpha * dv
        return x2, q

    def _apply_reverse_transform(self, x2: Tensor, v: Tensor, w: Tensor):
        cum_w = torch.cumsum(w, dim=-1)
        cum_w[..., -1] = 1.
        cum_w_inv = torch.cumsum((v[..., 1:] + v[..., :-1]) / 2. * w, dim=-1)
        cum_w_inv[..., -1] = 1.
        b = torch.clip(torch.searchsorted(cum_w_inv, torch.unsqueeze(x2, dim=-1).contiguous())[..., 0], max=cum_w_inv.shape[-1] - 1)
        v_ib, v_ibp1, w_ib = self._apply_indexing(v, w, b)
        dv = v_ibp1 - v_ib
        c0, c1, c2 = self._compute_coefficients(v, w, b, v_ib, w_ib, dv)
        alpha = self._solve_quadratic_equation(x2, c0, c1, c2)
        x2 = alpha * w_ib + (self._select_from_tensor(cum_w, b) - w_ib)
        q = v_ib + alpha * dv
        return x2, q

    def _compute_coefficients(self, v, w, b, v_ib, w_ib, dv):
        c0 = (v[..., :-1] + v[..., 1:]) / 2. * w
        c0 = self._select_from_tensor(torch.cumsum(c0, dim=-1), b) - self._select_from_tensor(c0, b)
        c1 = v_ib * w_ib
        c2 = dv * w_ib / 2.
        return c0, c1, c2

    def _select_from_tensor(self, t: Tensor, b):
        shape = t.shape
        idxs = self._get_dummy_indices(shape[:-1])
        idxs.append(b.view(-1))
        return t[idxs].view(*shape[:-1])

    def _apply_indexing(self, v: Tensor, w: Tensor, b: Tensor):
        v_ib = self._select_from_tensor(v, b)
        v_ibp1 = self._select_from_tensor(v, b + 1)
        w_ib = self._select_from_tensor(w, b)
        return v_ib, v_ibp1, w_ib

    def _solve_quadratic_equation(self, x2, c0, c1, c2):
        degenerate = (c2 == 0)
        if torch.any(degenerate):
            alpha = torch.zeros_like(x2)
            alpha[degenerate] = self._solve_degenerate(x2[degenerate], c0[degenerate], c1[degenerate])
            non_degenerate = ~ degenerate
            alpha[non_degenerate] = self._solve_non_degenerate(x2[non_degenerate], c0[non_degenerate], c1[non_degenerate], c2[non_degenerate])
        else:
            alpha = self._solve_non_degenerate(x2, c0, c1, c2)
        return alpha

    def _solve_degenerate(self, x2, c0, c1):
        return (x2 - c0) / c1

    def _solve_non_degenerate(self, x2, c0, c1, c2):
        c0_red= c0 - x2
        r = torch.sqrt(torch.pow(c1, 2.) - 4 * c0_red * c2)
        return torch.where(c1 >= 0, - 2. * c0_red / (c1 + r), (c1 - r) / (2. * c2))

    def _update_determinant_trackers(self, q):
        value = torch.mean(torch.log(torch.abs(q)), dim=list(range(1, len(q.shape))))
        dimensions = self.determinant_trackers[0].compute_dimensions(q, batch_dim=0)
        self.update_determinant_trackers(value, dimensions)

    def _get_dummy_indices(self, shape):
        return [x.flatten() for x in torch.meshgrid(*[torch.arange(s) for s in shape])]


if __name__ == '__main__':
    from torch import nn

    BATCH_SIZE = 10
    CHANNELS = 16
    DOMAIN_SIZE = (32, 64)

    K = 16

    data = torch.rand(BATCH_SIZE, CHANNELS, *DOMAIN_SIZE)

    print(data.min(), data.max())

    processor = nn.Sequential(
        nn.ReplicationPad2d((1, 1, 1, 1)),
        nn.Conv2d(CHANNELS // 2, CHANNELS // 2 * (2 * K + 1), (3, 3))
    )

    # processor = nn.Linear(CHANNELS // 2, CHANNELS // 2 * (2 * K + 1))

    model = RationalQuadraticSplineFlow(CHANNELS, processor, K)

    out = model.forward(data, direction=FlowDirection.FORWARD)
    reverted = model.forward(*out, direction=FlowDirection.REVERSE)


    print(torch.abs(data - reverted[0]).max())
