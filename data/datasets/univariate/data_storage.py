import argparse
import functools
import os.path

import common.utils
from common.utils import parse_range_string, is_valid_index_position

import pyrenderer


class VolumeDataStorage(object):

    FILE_PATTERN_TIME_KEY = 'time'
    FILE_PATTERN_ENSEMBLE_KEY = 'member'

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('VolumeDataStorage')
        example_pattern = ('"ScalarFlow/sim_{'
                             + VolumeDataStorage.FILE_PATTERN_ENSEMBLE_KEY
                             + ':06d}/density_{'
                             + VolumeDataStorage.FILE_PATTERN_TIME_KEY
                             + ':06d}.cvol"')
        prefix = '--data-storage:'
        group.add_argument(prefix + 'filename-pattern', type=str, help=f"""
        file name pattern used to generate volume file names. 
        To specify time and ensemble indices, the following keys are used:
        
        time: {VolumeDataStorage.FILE_PATTERN_TIME_KEY}
        ensemble: {VolumeDataStorage.FILE_PATTERN_ENSEMBLE_KEY}
        
        Example: {example_pattern}
        """, required=True)
        group.add_argument(prefix + 'base-path', type=str, default=None, help="""
        base path to be prepended in front of file name pattern
        """)
        group.add_argument(prefix + 'ensemble:index-range', type=str, default=None, help="""
            Ranges used for the ensemble index. The indices are obtained via
            <code>range(*map(int, {ensemble_index_range}.split(':')))</code>
            
            Example: "0:10:2" (Default: "0:1")
        """)
        group.add_argument(prefix + 'timestep:index-range', type=str, default=None, help="""
            Ranges used for the keyframes for time interpolation. 
            At those timesteps, representative vectors are generated, optionally trained,
            and interpolated between timesteps
            The indices are obtained via <code>range(*map(int, {timestep-index-range}.split(':')))</code>

            Example: "0:10:2" (Default: "0:1")
        """)
        group.add_argument(prefix + 'disable-file-verification', action='store_false', dest='verify_files_exist', help="""
        disable file verification for performance reasons
        """)
        group.set_defaults(verify_files_exist=True)

    def __init__(self, file_pattern, base_path=None, timestep_index=None, ensemble_index=None, verify_files_exist=True, verbose=False):
        self.base_path = os.path.abspath(base_path) if base_path is not None else None
        assert file_pattern is not None
        self.file_pattern = file_pattern
        if timestep_index is None:
            timestep_index = '0:1'
        self.timestep_index = parse_range_string(timestep_index)
        if ensemble_index is None:
            ensemble_index = '0:1'
        self.ensemble_index = parse_range_string(ensemble_index)
        if verify_files_exist:
            self._verify_files_exist()

    @classmethod
    def from_dict(cls, args):
        prefix = 'data_storage:'
        return cls(
            args[prefix + 'filename_pattern'], base_path=args[prefix + 'base_path'],
            timestep_index=args[prefix + 'timestep:index_range'], ensemble_index=args[prefix + 'ensemble:index_range'],
            verify_files_exist=args['verify_files_exist']
        )

    def _verify_files_exist(self):
        for idx_time in self.timestep_index:
            for idx_ensemble in self.ensemble_index:
                file_name = self._get_volume_file_name(idx_time, idx_ensemble)
                try:
                    self._load_volume_file(file_name)
                except Exception() as e:
                    raise Exception(f'[ERROR] Problem while loading volume file {file_name}: {e}')

    def _get_volume_file_name(self, idx_time, idx_ensemble):
        query = {
            VolumeDataStorage.FILE_PATTERN_TIME_KEY: idx_time,
            VolumeDataStorage.FILE_PATTERN_ENSEMBLE_KEY: idx_ensemble
        }
        file_name = self.file_pattern.format(**query)
        if self.base_path is not None:
            file_name = os.path.join(self.base_path, file_name)
        return file_name

    def num_timesteps(self):
        return len(self.timestep_index)

    def num_members(self):
        return len(self.ensemble_index)

    def num_ensembles(self):
        return self.num_members()

    @staticmethod
    @functools.lru_cache(4)
    def _load_volume_file(file_name):
        if not os.path.exists(file_name):
            raise ValueError(f'[ERROR] Volume file {file_name} does not exist.')
        vol = pyrenderer.Volume(file_name)
        return vol

    def load_volume(self, timestep=None, ensemble=None, index_access=False):
        if self.num_timesteps() > 1:
            assert timestep is not None, \
                f'[ERROR] VolumeDataStorage provides {self.num_timesteps()} timesteps. Argument <time> may not be None.'
        if timestep is None:
            timestep = 0 if index_access else self.timestep_index[0]
        if self.num_members() > 1:
            assert ensemble is not None, \
                f'[ERROR] VolumeDataStorage provides {self.num_members()} ensemble members. Argument <ensemble> may not be None.'
        if ensemble is None:
            ensemble = 0 if index_access else self.ensemble_index[0]
        if index_access:
            assert is_valid_index_position(timestep, self.timestep_index), f'[ERROR] Requested time index {timestep} is beyond admissible index range.'
            assert is_valid_index_position(ensemble, self.ensemble_index), f'[ERROR] Requested ensemble index {ensemble} is beyond admissible index range.'
            timestep = self.timestep_index[timestep]
            ensemble = self.ensemble_index[ensemble]
        else:
            assert timestep in self.timestep_index, f'[ERROR] Requested time step {timestep} is not available.'
            assert ensemble in self.ensemble_index, f'[ERROR] Requested ensemble member {ensemble} is not available.'
        file_name = self._get_volume_file_name(timestep, ensemble)
        return self._load_volume_file(file_name)

    def min_time(self):
        return min(self.timestep_index)

    def max_time(self):
        return max(self.timestep_index)


def _test_volume_data_storage():
    file_pattern = 'volumes/Ejecta/snapshot_070_256.cvol'
    base_path = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications'
    vds = VolumeDataStorage(
        file_pattern, base_path=base_path
    )
    volume = vds.load_volume()
    feature = volume.get_feature(0)
    level = feature.get_level(0).to_tensor()
    print('Finished')


if __name__ == '__main__':
    _test_volume_data_storage()