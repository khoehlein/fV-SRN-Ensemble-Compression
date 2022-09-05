from typing import Any, Tuple

import torch

from ..interface import InvertibleTransform, FlowDirection


class RandomPermute(InvertibleTransform):

    def __init__(self, channels: int):
        super(RandomPermute, self).__init__({
            FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1,
        })
        self.channels = channels
        self.register_buffer('permute', torch.randperm(channels))
        self.register_buffer('inverse_permute', torch.argsort(self.permute))

    def forward_transform(self, *data: Any) -> Tuple[Any, ...]:
        out = (data[0][..., self.permute],)
        return out

    def reverse_transform(self, *data: Any) -> Tuple[Any, ...]:
        out = (data[0][..., self.inverse_permute],)
        return out


if __name__ == '__main__':
    model = RandomPermute(3)
    data = torch.randn(10, 3)
    out = model.forward(data, FlowDirection.FORWARD)
    reverted = model.forward(*out, FlowDirection.REVERSE)

    print(data - reverted[0])