from typing import Optional, List

import torch
from pyrenderer import IVolumeInterpolation, Volume
from torch import Tensor

from data.datasets import OutputMode
from inference import IFieldEvaluator


class VolumeEvaluator(IFieldEvaluator):

    def __init__(self, interpolator: IVolumeInterpolation, device):
        super(VolumeEvaluator, self).__init__(3, 1, device)
        self.interpolator = interpolator
        try:
            self._default_volume = interpolator.volume()
            self._default_mipmap_level = interpolator.mipmap_level()
        except RuntimeError:
            self._default_volume = None
            self._default_mipmap_level = 0

    def forward(self, positions: Tensor) -> Tensor:
        return self.interpolator.evaluate(positions)

    def set_source(self, volume_data: Optional[Volume] = None, mipmap_level: Optional[int] = None, feature: Optional[str] = None):
        if volume_data is None:
            # assert self._default_volume is not None
            volume_data = self._default_volume
        if mipmap_level is None:
            mipmap_level = self._default_mipmap_level
        self.interpolator.setSource(volume_data, feature, mipmap_level)
        return self

    def restore_defaults(self):
        return self.set_source(self._default_volume, mipmap_level=self._default_mipmap_level)

    @staticmethod
    def output_mode():
        return OutputMode.DENSITY


class MultivariateVolumeEvaluator(IFieldEvaluator):

    def __init__(self, interpolator: IVolumeInterpolation, device):
        super(MultivariateVolumeEvaluator, self).__init__(3, None, device)
        self.interpolator = interpolator
        try:
            self._default_volume = interpolator.volume()
            self._default_mipmap_level = interpolator.mipmap_level()
        except RuntimeError:
            self._default_volume = None
            self._default_mipmap_level = 0
        self.restore_defaults()

    def forward(self, positions: Tensor) -> Tensor:
        out = []
        for feature_name in self._feature_names:
            self.interpolator.setSource(self._current_volume, feature_name, self._current_mipmap_level)
            out.append(self.interpolator.evaluate(positions))
        self.interpolator.setSource(self._default_volume, None, self._default_mipmap_level)
        return torch.cat(out, dim=-1)

    def _get_all_current_feature_names(self) -> List[str]:
        if self._current_volume is not None:
            return [
                self._current_volume.get_feature(i).name()
                for i in self._current_volume.num_features()
            ]
        else:
            return []

    def out_channels(self):
        return len(self._feature_names)

    def set_source(self, volume_data: Optional[Volume] = None, mipmap_level: Optional[int] = None, feature_names: Optional[List[str]] = None):
        if volume_data is None:
            volume_data = self._current_volume
        if volume_data is None:
            volume_data = self._default_volume
        assert volume_data is not None
        self._current_volume = volume_data
        if mipmap_level is None:
            mipmap_level = self._current_mipmap_level
        if mipmap_level is None:
            mipmap_level = self._default_mipmap_level
        assert mipmap_level is not None
        self._current_mipmap_level = mipmap_level
        all_feature_names = self._get_all_current_feature_names()
        if feature_names is None:
            feature_names = self._feature_names
        if feature_names is None:
            feature_names = all_feature_names
        assert len(set(feature_names).difference(set(all_feature_names))) == 0
        self._feature_names = feature_names
        return self

    def restore_defaults(self):
        self._current_volume = self._default_volume
        self._current_mipmap_level = self._default_mipmap_level
        self._feature_names = self._get_all_current_feature_names()

    def output_mode(self):
        return OutputMode.MULTIVARIATE


def _test():
    import pyrenderer
    vol = pyrenderer.Volume()
    vol.worldX, vol.worldY, vol.worldZ = 1, 1, 1
    vol.add_feature_from_tensor('a', torch.randn(1, 4, 4, 4))

    interpolator = pyrenderer.VolumeInterpolationGrid()
    interpolator.grid_resolution_new_behavior = True
    interpolator.setInterpolation(interpolator.Trilinear)

    evaluator = VolumeEvaluator(interpolator, device=torch.device('cuda:0'))
    evaluator.set_source(vol)
    evaluator.evaluate(torch.tensor([[.5, .5, .5]], device=torch.device('cuda:0')))
    evaluator.restore_defaults()
    print('Finished')


if __name__ == '__main__':
    _test()
