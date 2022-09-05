from torch import nn as nn


class ResidualConnection(nn.Module):

    def __init__(self, inner_module):
        super(ResidualConnection, self).__init__()
        self.inner_module = inner_module

    def forward(self, x):
        return x + self.inner_module(x)
