from typing import Any, Optional, Dict

import torch
from torch import Tensor, nn

from inference.model.latent_features.init import IInitializer


class PytorchInitializer(IInitializer):

    def __init__(self, method: str, method_kws: Optional[Dict[str, Any]] = None):
        if not method.endswith('_'):
            method = method+ '_'
        try:
            self.method = getattr(nn.init, method)
        except AttributeError:
            raise ValueError(f'[ERROR] Encountered invalid method specification. Module torch.nn.init does not provide a method {method}.')
        self.kws = method_kws if method_kws is not None else {}
        self._validate_method()

    def _validate_method(self):
        out = torch.zeros(2, 2)
        self.method(out, **self.kws)

    def get_tensor(self, *size: int, device=None, dtype=None) -> Tensor:
        out = torch.empty(*size, dtype=dtype, device=device)
        self.method(out, **self.kws)
        return out

    def init_tensor(self, x: Tensor) -> Tensor:
        self.method(x, **self.kws)
        return x


class Uniform(PytorchInitializer):

    def __init__(self, min=0., max=1.):
        super(Uniform, self).__init__('uniform_',method_kws={'a': min, 'b': max})


class SymmetricUniform(Uniform):

    def __init__(self, bound=1.):
        abs_bound = abs(bound)
        super(SymmetricUniform, self).__init__(min=-abs_bound, max=abs_bound)


class Normal(PytorchInitializer):

    def __init__(self, mean=0., std=1.):
        super(Normal, self).__init__('normal_', method_kws={'mean': mean, 'std': std})


Gaussian = Normal


def DefaultInitializer():
    return Normal(mean=0., std=0.01)
