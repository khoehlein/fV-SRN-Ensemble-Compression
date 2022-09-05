import math

import torch
from torch import nn, Tensor


class Sine(nn.Module):

    def __init__(self, omega=1):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x):
        return torch.sin(self.omega * x)


class Snake(nn.Module):

    def __init__(self, omega=1):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x: Tensor):
        return x + (1./self.omega) * (torch.sin(self.omega * x)**2)


class SnakeAlt(nn.Module):

    def __init__(self, omega=1.):
        super().__init__()
        self.omega = float(omega)

    def forward(self, x: Tensor):
        t = x + 1 - torch.cos(2. * self.omega * x)
        return t/(2. * self.omega)
