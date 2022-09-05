import os
import socket

import numpy as np
import torch
from torch import Tensor

import common.utils
import pyrenderer


DATA_BASE_PATH = {'host-name': 'path/to/data'}
FILE_NAME_NORM_KEY = 'norm_name'
FILE_NAME_VARIABLE_KEY = 'variable_name'
FILE_NAME_MEMBER_KEY = 'member'
FILE_NAME_TIME_KEY = 'time'

NORM_PATTERN = '{' + f'{FILE_NAME_NORM_KEY}' + '}_scaling'
VARIABLE_PATTERN = '{' + f'{FILE_NAME_VARIABLE_KEY}' + '}'
MEMBER_PATTERN = 'member{' + f'{FILE_NAME_MEMBER_KEY}' + ':04d}'
TIME_PATTERN = 't{' + f'{FILE_NAME_TIME_KEY}' + ':02d}.cvol'

VARIABLE_NAMES = ['tk', 'u', 'v', 'w', 'rh', 'qv', 'qhydro', 'z', 'dbz']


def get_data_base_path():
    host_name = socket.gethostname()
    return DATA_BASE_PATH[host_name]


def split_file_name_pattern(pattern):
    try:
        normpath = os.path.normpath(pattern)
        segments = normpath.split(os.sep)[-5:]
        assert segments[0] == 'single_variable'
        norm_code, variable_name, member_code, time_code = segments[1:]
        norm_name = norm_code.split('_')[0]
    except Exception:
        norm_name = None
        variable_name = None
    return norm_name, variable_name


def update_file_pattern_base_path(old_pattern):
    normpath = os.path.normpath(old_pattern)
    segments = normpath.split(os.sep)[-5:]
    assert segments[0] == 'single_variable'
    new_base_path = get_data_base_path()
    return os.path.join(new_base_path, *segments[1:])


def get_file_name_pattern(norm=None, variable=None, time=None, member=None, base_bath=True):
    file_path = []
    file_path.append(NORM_PATTERN.format(norm_name=norm) if norm is not None else NORM_PATTERN)
    file_path.append(VARIABLE_PATTERN.format(variable_name=variable) if variable is not None else VARIABLE_PATTERN)
    file_path.append(MEMBER_PATTERN.format(member=member) if member is not None else MEMBER_PATTERN)
    file_path.append(TIME_PATTERN.format(time=time) if time is not None else TIME_PATTERN)
    file_path = os.path.join(*file_path)
    if base_bath:
        file_path = os.path.join(get_data_base_path(), file_path)
    return file_path


def load_ensemble(norm, variable, time=4, min_member=1, max_member=65):
    file_name_pattern = get_file_name_pattern(norm=norm, variable=variable, time=time)
    data = []
    for member in range(min_member, max_member):
        file_name = file_name_pattern.format(member=member)
        vol = pyrenderer.Volume(file_name)
        data.append(vol.get_feature(0).get_level(0).to_tensor()[0].data.cpu().numpy())
    return data


def load_scales(norm, variable):
    scales_file = os.path.join(
        get_data_base_path(),
        NORM_PATTERN.format(norm_name=norm),
        VARIABLE_PATTERN.format(variable_name=variable),
        'scales.npz'
    )
    scales_data = np.load(scales_file)
    offset = scales_data['offset']
    scale = scales_data['scale']
    return offset, scale


def revert_scaling(data, scales):
    offset, scale = scales
    assert len(data.shape) == 4
    if len(offset.shape) < 4:
        offset = offset[None, ...]
        scale = scale[None, ...]
    return data * scale + offset


def get_sampling_positions(resolution):
    positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
    positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
    return positions


class SingleVariableData(object):

    def __init__(self, variable_name: str, normalization: str, device=None, dtype=None):
        self.variable_name = variable_name
        self.normalization = normalization
        self.device = torch.device('cpu') if device is not None else device
        self.dtype = dtype

    def get_variable_base_path(self):
        return os.path.join(
            get_data_base_path(),
            NORM_PATTERN.format(norm_name=self.normalization),
            VARIABLE_PATTERN.format(variable_name=self.variable_name)
        )

    def load_volume(self, member: int, timestep: int):
        path = os.path.join(
            self.get_variable_base_path(),
            MEMBER_PATTERN.format(member=member),
            TIME_PATTERN.format(time=timestep),
        )
        return pyrenderer.Volume(path)

    def _postprocess(self, data: Tensor):
        if data.device != self.device:
            data = data.to(self.device)
        if self.dtype is not None and data.dtype != self.dtype:
            data = data.to(self.dtype)
        return data

    def load_tensor(self, member: int, timestep: int):
        volume = self.load_volume(member, timestep)
        data = volume.get_feature(0).get_level(0).to_tensor()
        return self._postprocess(data)

    def load_scales(self, return_volume=False):
        scales_file = os.path.join(self.get_variable_base_path(), 'scales.npz')
        scales_data = np.load(scales_file)
        offset = torch.from_numpy(scales_data['offset'][None, ...])
        scale = torch.from_numpy(scales_data['scale'][None, ...])
        offset = self._postprocess(offset)
        scale = self._postprocess(scale)
        if return_volume:
            vol = pyrenderer.Volume()
            vol.worldX = 10.
            vol.worldY = 10.
            vol.worldZ = 1.
            vol.add_feature_from_tensor('offset', offset.to(torch.float32).cpu())
            vol.add_feature_from_tensor('scale', scale.to(torch.float32).cpu())
            return vol
        return offset, scale


def _test():
    data = SingleVariableData('tk', 'level-min-max')
    volume = data.load_tensor(1, 4)
    scales = data.load_scales(return_volume=True)
    print('Finished')


if __name__ == '__main__':
    _test()
