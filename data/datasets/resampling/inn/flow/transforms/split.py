from typing import List, Optional, Tuple
from math import fsum
import torch
from torch import Tensor
from ..interface import InvertibleTransform, FlowDirection


class _SplitOrMerge(InvertibleTransform):

    def __init__(self, channels: int, sections: List[int], direction_of_split: FlowDirection, dim: Optional[int]=1, label: Optional[str] = None):
        assert type(sections) == list
        for s in sections:
            assert int(s) == s, f'[ERROR] Cannot interpret input of type {type(s)} as int'
        assert channels % fsum(sections) == 0
        self.channels = channels
        self.sections = tuple(int((channels * s) // fsum(sections)) for s in sections)
        self.dim = dim
        self.direction = direction_of_split
        self._transform_mapping = {
            self.direction: self._split_sections,
            self.direction.opposite(): self._merge_sections,
        }
        super(_SplitOrMerge, self).__init__({
            self.direction: 1,
            self.direction.opposite(): len(sections),
        }, label=label)

    def _split_sections(self, x: Tuple[Tensor]) -> Tuple[Tensor]:
        assert len(x) == 1, \
            f'[ERROR] Split function expected input tuple of length 1 but got length {len(x)}.'
        x = x[0]
        channels = x.shape[self.dim]
        assert channels == self.channels
        return tuple(torch.split(x, self.sections, dim=self.dim))

    def _merge_sections(self, x: Tuple[Tensor]) -> Tuple[Tensor]:
        x_sections = tuple(int(t.shape[self.dim]) for t in x)
        assert self.sections == x_sections, \
            f'[ERROR] Join function expected sections of size {self.sections} but got {x_sections}.'
        out = (torch.cat(x, dim=self.dim),)
        return out

    def forward_transform(self, *data: Tensor) -> Tuple[Tensor]:
        transform = self._transform_mapping[FlowDirection.FORWARD]
        return transform(data)

    def reverse_transform(self, *data: Tensor) -> Tuple[Tensor]:
        transform = self._transform_mapping[FlowDirection.REVERSE]
        return transform(data)


class Split(_SplitOrMerge):

    def __init__(self, channels: int, sections: List[int], dim: Optional[int]=1, label: Optional[str] = None):
        super(Split, self).__init__(channels, sections, FlowDirection.FORWARD, dim=dim, label=label)


class Merge(_SplitOrMerge):

    def __init__(self, channels: int, sections: List[int], dim: Optional[int] = 1, label: Optional[str] = None):
        super(Merge, self).__init__(channels, sections, FlowDirection.REVERSE, dim=dim, label=label)
