from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from ..interface import InvertibleTransform, FlowDirection


class Gaussianize(InvertibleTransform):

    def __init__(self, channels, split: Optional[bool] = True, label: Optional[str] = None):
        super(Gaussianize, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        self.split = split
        in_channels = channels // 2 if split else channels
        self.parameterize = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, (3, 3), padding=(1, 1)),
            nn.Conv2d(2 * in_channels, 2 * in_channels, (1, 1))
        )
        for i in range(2):
            # nn.init.normal_(self.parameterize[i].weight.data, 0., 1.e-6)
            # nn.init.normal_(self.parameterize[i].bias.data, 0., 1.e-6)
            nn.init.zeros_(self.parameterize[i].weight.data)
            nn.init.zeros_(self.parameterize[i].bias.data)

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        if self.split:
            x1, x2 = torch.chunk(data[0], 2, dim=1)
        else:
            x1, x2 = torch.zeros_like(data[0]), data[0]
        params = self.parameterize(x1)
        mus, log_scales = params[:, ::2, ...], params[:, 1::2, ...]
        x2 = (x2 - mus) * torch.exp(- log_scales)
        if self.has_determinant_tracker():
            self._update_determinant_trackers(log_scales)
        if self.split:
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x2
        out = (out,)
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        if self.split:
            x1, x2 = torch.chunk(data[0], 2, dim=1)
        else:
            x1, x2 = torch.zeros_like(data[0]), data[0]
        params = self.parameterize(x1)
        mus, log_scales = params[:, ::2, ...], params[:, 1::2, ...]
        x2 = torch.exp(log_scales) * x2 + mus
        if self.has_determinant_tracker():
            self._update_determinant_trackers(log_scales)
        if self.split:
            out = torch.cat([x1, x2], dim=1)
        else:
            out = x2
        out = (out,)
        return out

    def _update_determinant_trackers(self, log_scales: Tensor):
        shape = tuple(log_scales.shape)
        value = torch.mean(log_scales, dim=tuple(range(1, len(shape))))
        dimensions = self.determinant_trackers[0].compute_dimensions(shape=shape, batch_dim=0)
        self.update_determinant_trackers(value, dimensions)


if __name__ == '__main__':
    data = torch.randn(10, 16, 8, 16)
    flow = Gaussianize(channels=16,split=False)
    code = flow(data, direction=FlowDirection.FORWARD)
    reverted = flow.forward(*code, direction=FlowDirection.REVERSE)[0]
    print(data[0, 0, :4, :4])
    print(reverted[0, 0, :4, :4])
    print(data[0, 8, :4, :4])
    print(reverted[0, 8, :4, :4])