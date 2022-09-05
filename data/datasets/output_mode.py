from enum import Enum


class OutputMode(Enum):
    DENSITY = 'density'
    RGBO = 'rgbo'
    MULTIVARIATE = 'multivariate'


class MultivariateOutputMode(object):

    def __init__(self, num_channels):
        self.num_channels = num_channels

    # class MULTIVARIATE(object):
    #
    #     def __init__(self, d: int):
    #         self.value = 'multivariate'
    #         self.d = d