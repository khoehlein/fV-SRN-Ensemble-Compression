from typing import Any, Optional
import torch
import torch.nn as nn
from math import log
from numpy import pi as PI


class ArctanClamping(nn.Module):

    def __init__(self, clamp: Optional[float]=2., log_scale: Optional[bool]=True):
        super(ArctanClamping, self).__init__()
        assert clamp >= (1.if log_scale else 0.)
        self.clamp = log(clamp) if log_scale else clamp

    def forward(self, data: Any) -> Any:
        scale = (2. * self.clamp / PI)
        return scale * torch.arctan(data / scale)
