from enum import Enum


class BackendOutputMode(Enum):
    DENSITY = 'density'
    DENSITY_DIRECT = 'density:direct'
    RGBO = 'rgbo'
    RGBO_DIRECT = 'rgbo:direct'
    RGBO_EXP = 'rgbo:exp'
    MULTIVARIATE = 'multivariate'