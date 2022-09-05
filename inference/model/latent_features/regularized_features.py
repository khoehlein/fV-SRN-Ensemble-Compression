from typing import Any, Optional

import torch
from torch import Tensor, nn

from .interface import ILatentFeatures


class RegularizedFeatures(ILatentFeatures):

    def __init__(self, feature_module: ILatentFeatures, sparsity_weight: Optional[float] = 0.):
        super(RegularizedFeatures, self).__init__(feature_module.dimension, feature_module.num_channels(),debug=feature_module.is_debug())
        self.feature_module = feature_module.set_debug(False)
        self.register_parameter('regularization', nn.Parameter(torch.ones(feature_module.num_channels()), requires_grad=True))
        self.sparsity_weight = sparsity_weight

    def forward(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        out = self.feature_module.evaluate(positions, time, member)
        return self.regularization[None, :] * torch.tanh(out)

    def reset_member_features(self, *member_keys: Any) -> 'ILatentFeatures':
        self.feature_module.reset_member_features(*member_keys)
        return self

    def num_key_times(self) -> int:
        return self.feature_module.num_key_times()

    def num_members(self) -> int:
        return self.feature_module.num_members()

    def uses_positions(self) -> bool:
        return self.feature_module.uses_positions()

    def uses_member(self) -> bool:
        return self.feature_module.uses_positions()

    def uses_time(self) -> bool:
        return self.feature_module.uses_time()

    def compute_regularization(self):
        return self.sparsity_weight * torch.sum(torch.abs(self.regularization))

    def num_active_channels(self, eps=1.e-6):
        return torch.sum((torch.abs(self.regularization) >= eps).to(torch.int32)).item()
