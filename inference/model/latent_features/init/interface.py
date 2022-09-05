from torch import Tensor


class IInitializer(object):

    def get_tensor(self, *shape: int, device=None, dtype=None) -> Tensor:
        raise NotImplementedError()

    def init_tensor(self, x: Tensor) -> Tensor:
        raise NotImplementedError()