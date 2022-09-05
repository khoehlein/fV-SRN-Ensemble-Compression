import struct
import warnings
from typing import Union, Optional

import numpy as np
import pyrenderer
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class VolumetricScene(object):

    class VolumetricFeature(nn.Module):

        def __init__(self, data: Tensor, name: Optional[str] = None):
            super(VolumetricScene.VolumetricFeature, self).__init__()
            assert len(data.shape) == 4, '[ERROR] Feature data tensor must be 4D with dimensions (C, X, Y, Z)'
            self.name = name
            self.register_buffer('data', data)

        def grid_size(self):
            # grid size is returned as (X, Y, Z)
            return tuple(self.data.shape[1:])

        def num_channels(self):
            return self.data.shape[0]

    class CVOLFile(object):
        """
        Reader class for loading CVOL files to pytorch tensors

        CVOL description:
         * \brief The main storage class for volumetric datasets.
         *
         * The volume stores multiple feature channels,
         * where each feature describes, e.g., density, velocity, color.
         * Each feature is specified by the number of channel and the data type
         * (unsigned char, unsigned short, float)
         * The feature channels can have different resolutions, but are all
         * mapped to the same bounding volume.
         *
         * Format of the .cvol file (binary file):
         * <pre>
         *  [64 Bytes Header]
         *   4 bytes: magic number "CVOL"
         *   4 bytes: version (int)
         *   3*4 bytes: worldX, worldY, worldZ of type 'float'
         *     the size of the bounding box in world space
         *   4 bytes: number of features (int)
         *   4 bytes: flags (int), OR-combination of \ref Volume::Flags
         *	 4 bytes: unused
         *  [Content, repeated for the number of features]
         *   4 bytes: length of the name (string)
         *	 n bytes: the contents of the name string (std::string)
         *	 3*8 bytes: sizeX, sizeY, sizeZ of type 'uint64_t'
         *	   the voxel resolution of this feature
         *	 4 bytes: number of channels (int)
         *	 4 bytes: datatype (\ref DataType)
         *   Ray memory dump of the volume, sizeX*sizeY*sizeZ*channels entries of type 'datatype'.
         *     channels is fastest, followed by X and Y, Z is slowest.
         * </pre>
         *
         * Legacy format, before multi-channel support was added:
         * Format of the .cvol file (binary file):
         * <pre>
         *  [64 Bytes Header]
         *   4 bytes: magic number "cvol"
         *   3*8 bytes: sizeX, sizeY, sizeZ of type 'uint64_t'
         *   3*8 bytes: voxel size X, Y, Z in world space of type' double'
         *   4 bytes: datatype, uint value of the enum \ref Volume::DataType
         *	 1 byte: bool if the volume contents are LZ4-compressed
         *   7 bytes: padding, unused
         *  [Content]
         *
         *   Ray memory dump of the volume, sizeX*sizeY*sizeZ entries of type 'datatype'.
         *   X is fastest, Z is slowest.
        """

        @staticmethod
        def _uint_from_bytes(bytes):
            return int.from_bytes(bytes, 'little', signed=False)

        @staticmethod
        def _float_from_bytes(bytes):
            return struct.unpack('f', bytes)[0]

        @staticmethod
        def _double_from_bytes(bytes):
            return struct.unpack('d', bytes)[0]

        def __init__(self, file_name: str):
            assert file_name.endswith('.cvol')
            self.file_name = file_name

        def export_scene_data(self):
            with open(self.file_name, 'rb') as f:
                code = f.read(4).decode()
                if code == 'cvol':
                    scene_data = self._process_old_file_format(f)
                elif code == 'CVOL':
                    scene_data = self._process_new_file_format(f)
                else:
                    raise Exception(
                        f'[ERROR] Encountered unknown magic code {code}. Something seems to have gone wrong.')
                assert f.read(
                    1) == b'', '[ERROR] Encountered trailing file contents, for which no interpretation is available'
            return scene_data

        def _process_new_file_format(self, f):
            scene_data, num_features = self._read_new_header(f)
            for _ in range(num_features):
                data, name = self._read_new_feature(f)
                scene_data.add_feature(data, name=name)
            return scene_data

        def _read_new_feature(self, f):
            length_of_name = self._uint_from_bytes(f.read(4))
            if length_of_name > 0:
                name = f.read(length_of_name).decode()
            else:
                name = None
            grid_size = tuple(reversed([self._uint_from_bytes(f.read(8)) for _ in range(3)]))
            num_channels = self._uint_from_bytes(f.read(4))
            data_type_code = self._uint_from_bytes(f.read(4))
            data = self._read_contents(f, grid_size + (num_channels,), data_type_code)
            data = self._apply_postprocessing(data, data_type_code)
            return data, name

        def _read_new_header(self, f):
            version = self._uint_from_bytes(f.read(4))
            world_size = tuple([self._float_from_bytes(f.read(4)) for _ in range(3)])
            num_features = self._uint_from_bytes(f.read(4))
            flags = self._uint_from_bytes(f.read(4))
            f.read(4)
            scene_data = VolumetricScene(version=version, size=world_size, flags=flags)
            return scene_data, num_features

        def _process_old_file_format(self, f):
            scene_data, grid_size, data_type_code = self._read_old_header(f)
            assert scene_data.flags == 0, f'[ERROR] CVOL Reader does currently only support flag 0. Got {scene_data.flags} instead.'
            data = self._read_contents(f, grid_size + (1,), data_type_code)
            data = self._apply_postprocessing(data, data_type_code)
            scene_data.add_feature(data)
            return scene_data

        def _apply_postprocessing(self, data, data_type_code):
            if data_type_code == 0:
                data = data / 255.
            return data

        def _read_old_header(self, f):
            grid_size = [self._uint_from_bytes(f.read(8)) for _ in range(3)]
            voxel_size = [self._double_from_bytes(f.read(8)) for _ in range(3)]
            world_size = tuple([g * v for g, v in zip(grid_size, voxel_size)])
            data_type_code = self._uint_from_bytes(f.read(4))
            flags = int(f.read(1) != b'\x00')
            f.read(7)
            scene_data = VolumetricScene(version=None, size=world_size, flags=flags)
            return scene_data, tuple(reversed(grid_size)), data_type_code

        def _read_contents(self, f, grid_size, data_type_code):
            bytes_per_value = 2 ** data_type_code
            buffer = f.read(bytes_per_value * np.prod(grid_size))
            format = {0: 'B', 1: 'H', 2: 'f'}
            contents = memoryview(buffer).cast(format[data_type_code])
            data = torch.tensor(contents).view(grid_size).T
            return data

    def __init__(self, version=None, size=None, flags=None):
        self.version = version
        if size is None:
            size = (1., 1., 1.)
        assert type(size) == tuple and len(size) == 3 and np.all([float(i) == i for i in size])
        self.size = size
        if flags is None:
            flags = 0
        self.flags = flags
        self._volumetric_features = []
        self._features_by_name = {}
        self._active_feature = None

    def world_size(self):
        # world size is returned as (worldZ, worldY, worldX)!
        return self.size

    @classmethod
    def from_cvol(cls, file_name: str):
        return cls.CVOLFile(file_name).export_scene_data()

    def num_features(self):
        return len(self._volumetric_features)

    def set_active_feature(self, id: Union[str, int]):
        if type(id) == int:
            feature = self._volumetric_features[id]
        elif type(id) == str:
            feature = self._features_by_name[id]
        else:
            raise Exception('[ERROR] Got invlid feature identifier. Identifier must be int or string.')
        self._active_feature = feature
        return self

    def get_active_feature(self):
        if self._active_feature is None:
            if self.num_features() == 1:
                self.set_active_feature(0)
            else:
                raise Exception('[ERROR] Scene data has multiple features and none has been selected as active.' )
        return self._active_feature

    def add_feature(self, data: Tensor, name: Optional[str] = None):
        feature = VolumetricScene.VolumetricFeature(data, name=name)
        self._volumetric_features.append(feature)
        if feature.name is not None:
            self._features_by_name[feature.name] = feature
        return self

    def export_to_pyrenderer(self):
        vol = pyrenderer.Volume()
        vol.worldX, vol.worldY, vol.worldZ = self.size
        counter_unnamed = 0
        for feature in self._volumetric_features:
            name = feature.name
            if name is None:
                name = f'unnamed:{counter_unnamed}'
                counter_unnamed = counter_unnamed + 1
            data = feature.data.to(device='cpu', dtype=torch.float32)
            vol.add_feature_from_tensor(name, data)
        return vol

    def save(self, file_name: str, compression=0):
        vol = self.export_to_pyrenderer()
        if compression > 0:
            self._raise_compression_warning()
        vol.save(file_name,compression=compression)
        return self

    @staticmethod
    def _raise_compression_warning():
        message = '[WARN] Encountered non-zero compression during saving. '
        message += 'Saving should work but compressed cvol files can currently not be loaded!'
        warnings.warn(message)


class VolumeInterpolation(nn.Module):

    def __init__(self, method='bilinear', align_corners=True):
        super(VolumeInterpolation, self).__init__()
        self.method = method
        self.align_corners = align_corners

    def interpolate(self, positions: Tensor, feature: VolumetricScene.VolumetricFeature):
        shape = positions.shape
        grid = (2. * torch.flip(positions, [-1]) - 1.).view(1, 1, 1, *shape)
        num_channels = feature.num_channels()
        data = feature.data[None, ...]
        out = F.grid_sample(data, grid, mode=self.method, align_corners=self.align_corners, padding_mode='border')
        return out.view(shape[0], num_channels)


def _test_reader():
    import os

    applications_root = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications'
    settings_file = 'config-files/meteo-ensemble-t_0-6-m_1-10.json'
    data_file = 'volumes/Ejecta/snapshot_070_256.cvol'

    settings_file, data_file =[os.path.join(applications_root, f) for f in [settings_file, data_file]]

    device = torch.device('cuda:0')
    # get positions of tensor corners
    positions = torch.meshgrid(*([torch.linspace(0., 1., 2, device=device)] * 3))
    positions = torch.stack(positions, dim=-1).view(-1, 3)

    # load scene with own interface
    scene = VolumetricScene.from_cvol(data_file)
    feature = scene.get_active_feature().to(device)

    # get tensor data
    index_corners = torch.unbind(-positions.to(dtype=torch.long), dim=-1)
    values_tensor = feature.data[0][index_corners]

    # setting 1: original pyrenderer loading and interpolation
    interpolator = pyrenderer.VolumeInterpolationGrid()
    interpolator.setInterpolation(interpolator.Trilinear)
    volume_pyrenderer = pyrenderer.Volume(data_file)
    interpolator.setSource(volume_pyrenderer, 0)
    values_pyrenderer = interpolator.evaluate(positions).view(-1)

    #setting 2: load with VolumetricScene, interpolate with pyrenderer
    volume_hybrid = scene.export_to_pyrenderer()
    interpolator.setSource(volume_hybrid, 0)
    values_hybrid = interpolator.evaluate(positions).view(-1)

    # setting 3: load with VolumetricScene, interpolate with VolumeInterpolation
    interpolator = VolumeInterpolation(method='bilinear', align_corners=True)
    values_scene = interpolator.interpolate(positions, feature).view(-1)

    # setting 4: load with pyrenderer, interpolate with pyrenderer from settings
    image_evaluator = pyrenderer.load_from_json(settings_file)
    interpolator = image_evaluator.volume
    interpolator.setSource(volume_pyrenderer, 0)
    values_pyrenderer_with_settings = interpolator.evaluate(positions).view(-1)

    print('Finished interpolation')

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex='all', sharey='all')
    fig.suptitle('method vs. tensor')
    ax = ax.ravel()
    for i, (title, vals) in enumerate([
        ('original', values_pyrenderer), ('original w. settings', values_pyrenderer_with_settings),
        ('hybrid', values_hybrid), ('scene', values_scene)
    ]):
        ax[i].plot([0, 0.035], [0, 0.035])
        ax[i].scatter(values_tensor.data.cpu().numpy(), vals.data.cpu().numpy())
        ax[i].set(title=title)
    plt.tight_layout()
    plt.show()
    plt.close()

def _get_tensors():
    file_name = '....cvol'
    volume = pyrenderer.Volume(file_name)
    tensors = []
    for i in range(volume.num_features()):
        feature = volume.get_feature(i)
        level = feature.get_level(0)
        tensors.append(level.to_tensor())



if __name__== '__main__':
    data = VolumetricScene.from_cvol('/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/members/tk/t02.cvol')
    data.set_active_feature(0)
    feature = data.get_active_feature()
    print('Finished')