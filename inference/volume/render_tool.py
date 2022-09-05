import argparse
import os.path
from typing import Callable

import pyrenderer

from common import utils
from pyrenderer import IImageEvaluator

from .volume_evaluator import VolumeEvaluator


class RenderTool(object):

    _config_file_mapper = None

    @staticmethod
    def set_config_file_mapper(f: Callable[[str], str]):
        """
        Specifies a function that maps config file names from the input settings
        to the actual filename to be loaded.
        This is used to translate from server-side filenames to user-side filenames
        when loading pre-trained models.
        :param f: the callable [str]->str
        """
        RenderTool._config_file_mapper = f

    @staticmethod
    def _map_config_file(filename: str):
        if RenderTool._config_file_mapper is None:
            return filename
        return RenderTool._config_file_mapper(filename)

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('Rendering')
        prefix = '--renderer:'
        group.add_argument(prefix + 'settings-file', type=str, required=True, help="""
            Settings .json file,specifying camera, step size and initial volume.
        """)
        group.add_argument(prefix + 'tf-dir', type=str, default=None, help="""
        Directory with transfer function files for TF-generalization training.
        If not specified, the TF from the settings is used.
        If specified, this replaces the TF from the settings.
        """)

    @classmethod
    def from_dict(cls, args, device):
        settings_file = os.path.abspath(args['renderer:settings_file'])
        if cls._config_file_mapper is not None:
            settings_file = cls._config_file_mapper(settings_file)
        image_evaluator = pyrenderer.load_from_json(settings_file)
        return cls(image_evaluator, device, settings_file=settings_file)

    def __init__(self, image_evaluator: IImageEvaluator, device, tf_directory=None, settings_file=None):
        self.image_evaluator = image_evaluator
        self._tf_directory = tf_directory # currently not used
        try:
            self._default_volume = self.image_evaluator.volume.volume()
            self._default_mipmap_level = self.image_evaluator.volume.mipmap_level()
        except RuntimeError:
            self._default_volume = None
            self._default_mipmap_level = 0
        self._default_camera_pitch_yaw_distance = utils.copy_double3(image_evaluator.camera.pitchYawDistance.value)
        self._default_settings_file = settings_file
        self.device = device

    def default_camera_pitch_yaw_distance(self):
        return self._default_camera_pitch_yaw_distance

    def set_source(self, volume_data, mip_map_level=None, feature=None):
        if mip_map_level is None:
            mip_map_level = self._default_mipmap_level
        if feature is None:
            feature = volume_data.get_feature(0).name()
        self.image_evaluator.volume.setSource(volume_data, feature, mip_map_level)
        return self

    def restore_defaults(self):
        self.image_evaluator.volume.setSource(self._default_volume, None, self._default_mipmap_level)
        self.image_evaluator.camera.pitchYawDistance.value = self._default_camera_pitch_yaw_distance
        return self

    def get_volume_evaluator(self):
        return VolumeEvaluator(self.image_evaluator.volume, self.device)

    def get_image_evaluator(self):
        return self.image_evaluator

    def get_settings_file(self):
        return self._default_settings_file