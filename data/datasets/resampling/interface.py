from typing import Union

import torch

from inference import IFieldEvaluator
from .coordinate_box import CoordinateBox, UnitCube


class IImportanceSampler(object):

    def __init__(self, dimension: int, root_box: Union[CoordinateBox, None], device):
        self.dimension = dimension
        if root_box is None:
            root_box = UnitCube(dimension, device=device)
        else:
            assert root_box.dimension == dimension
            assert root_box.device == device
        self.root_box = root_box
        if device is None:
            device = torch.device('cpu')
        self.device = device

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator, **kwargs):
        raise NotImplementedError()
