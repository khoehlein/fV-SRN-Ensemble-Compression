from typing import Any, Optional

import torch
from torch import Tensor

from inference.model.latent_features.interface import ILatentFeatures, IFeatureModule
from inference.model.latent_features.marginal.ensemble_features import IEnsembleFeatures
from inference.model.latent_features.marginal.temporal_features import ITemporalFeatures


class MarginalLatentFeatures(ILatentFeatures):

    def uses_member(self) -> bool:
        return self.ensemble_features is not None

    def uses_time(self) -> bool:
        return self.temporal_features is not None

    def __init__(
            self,
            temporal_features: Optional[ITemporalFeatures] = None,
            ensemble_features: Optional[IEnsembleFeatures] = None,
            volumetric_features: Optional[IFeatureModule] = None,
    ):
        dimension = None
        if dimension is None and temporal_features is not None:
            dimension = temporal_features.dimension
        if dimension is None and ensemble_features is not None:
            dimension = ensemble_features.dimension
        if dimension is None and volumetric_features is not None:
            dimension = volumetric_features.dimension
        assert dimension is not None, '[ERROR] At least one of temporal, ensemble or volumetric features must not be None.'
        num_features = 0
        debug = False
        if temporal_features is not None:
            assert temporal_features.dimension == dimension
            assert temporal_features.num_channels() > 0
            num_features = num_features + temporal_features.num_channels()
            debug = temporal_features.is_debug() or debug
            temporal_features.set_debug(False)
        if ensemble_features is not None:
            assert ensemble_features.dimension == dimension
            assert ensemble_features.num_channels() > 0
            num_features = num_features + ensemble_features.num_channels()
            debug = ensemble_features.is_debug() or debug
            ensemble_features.set_debug(False)
        if volumetric_features is not None:
            assert volumetric_features.dimension == dimension
            assert volumetric_features.num_channels() > 0
            num_features = num_features + volumetric_features.num_channels()
            debug = volumetric_features.is_debug() or debug
            volumetric_features.set_debug(False)
        super(MarginalLatentFeatures, self).__init__(dimension, num_features, debug)
        self.temporal_features = temporal_features
        self.ensemble_features = ensemble_features
        self.volumetric_features = volumetric_features

    def forward(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        features = []
        if self.temporal_features is not None:
            features.append(self.temporal_features.evaluate(positions, time, member))
        if self.ensemble_features is not None:
            features.append(self.ensemble_features.evaluate(positions, time, member))
        if self.volumetric_features is not None:
            features.append(self.volumetric_features.evaluate(positions, time, member))
        out = torch.cat(features, dim=-1)
        return out

    def reset_member_features(self, *member_keys: Any) -> 'MarginalLatentFeatures':
        self.ensemble_features.reset_member_features(*member_keys)
        return self

    def num_key_times(self) -> int:
        if self.temporal_features is None:
            return 0
        return self.temporal_features.num_key_times()

    def num_members(self) -> int:
        if self.ensemble_features is None:
            return 0
        return self.ensemble_features.num_members()

    def uses_positions(self):
        if self.volumetric_features is not None:
            return True
        if self.temporal_features is not None and self.temporal_features.uses_positions():
            return True
        if self.ensemble_features is not None and self.ensemble_features.uses_positions():
            return True
        return False

    def uses_linear_features(self):
        if self.temporal_features is not None and not self.temporal_features.uses_positions():
            return True
        if self.ensemble_features is not None and not self.ensemble_features.uses_positions():
            return True
        return False
