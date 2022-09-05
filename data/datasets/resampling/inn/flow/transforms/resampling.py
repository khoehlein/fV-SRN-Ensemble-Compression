from typing import Any, Optional, Tuple,Union
from numbers import Number
import torch
from torch import Tensor
from numpy import prod
from ..interface import InvertibleTransform, FlowDirection


class _Resampling2d(InvertibleTransform):

    def __init__(
            self,
            in_channels: int, out_channels: int,
            direction_to_lower: FlowDirection,
            patch_size: Optional[Union[Number, Tuple[Number, ...]]] = (2, 2),
            label: Optional[str] = None
    ):
        super(_Resampling2d, self).__init__({FlowDirection.FORWARD: 1, FlowDirection.REVERSE: 1}, label=label)
        self.patch_size = None
        self._parse_patch_size(patch_size)
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._transform_mapping = {
            direction_to_lower: self._to_lower_resolution,
            direction_to_lower.opposite(): self._to_higher_resolution,
        }

    def _parse_patch_size(self, patch_size: Any):
        if isinstance(patch_size, Number):
            assert int(patch_size) == patch_size, \
                f'[ERROR] Patch size of type {type(patch_size)} cannot be interpreted as int.'
            patch_size = (patch_size, patch_size)
        else:
            assert len(patch_size) == 2, \
                '[ERROR] Patch size must be given as int or sized object of length 2.'
        for p in patch_size:
            assert int(p) == p, \
                f'[ERROR] Patch size of type {type(p)} cannot be interpreted as int.'
        self.patch_size = tuple(int(p) for p in patch_size)

    def _to_lower_resolution(self, *x: Tensor) -> Tensor:
        assert len(x) == 1, f'[ERROR] Resampling to lower expected 1 input tensor but got {len(x)} instead'
        x = x[0]
        x = torch.stack(torch.split(x, self.patch_size[0], dim=2), dim=-1)
        x = torch.stack(torch.split(x, self.patch_size[1], dim=3), dim=-1)
        x = torch.flatten(x, start_dim=1, end_dim=3)
        return x

    def _to_higher_resolution(self, *x: Tensor) -> Tensor:
        assert len(x) == 1, f'[ERROR] Resampling to higher expected 1 input tensor but got {len(x)} instead'
        x = x[0]
        shape = x.shape
        new_shape = (
            shape[0],
            shape[1] // (self.patch_size[0] * self.patch_size[1]),
            self.patch_size[0], self.patch_size[1],
            shape[2], shape[3]
        )
        x = torch.reshape(x, new_shape)
        x = torch.cat(torch.split(x, 1, dim=-1), dim=3).squeeze(dim=-1)
        x = torch.cat(torch.split(x, 1, dim=4), dim=2).squeeze(dim=-1)
        return x

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        transform = self._transform_mapping[FlowDirection.FORWARD]
        out = (transform(*data),)
        return out

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor, ...]:
        transform = self._transform_mapping[FlowDirection.REVERSE]
        out = (transform(*data),)
        return out


class Subsampling2d(_Resampling2d):

    def __init__(self, channels: int, patch_size: Optional[Union[Number, Tuple[Number, ...]]] = (2, 2), label: Optional[str] = None):
        super(Subsampling2d, self).__init__(
            channels, channels * prod(patch_size), FlowDirection.FORWARD, patch_size=patch_size, label=label
        )


class Supersampling2d(_Resampling2d):

    def __init__(self, channels: int, patch_size: Optional[Union[Number, Tuple[Number, ...]]] = (2, 2), label: Optional[str] = None):
        super(Supersampling2d, self).__init__(
            channels, channels // prod(patch_size), FlowDirection.REVERSE, patch_size=patch_size, label=label
        )
