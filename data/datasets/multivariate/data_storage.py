import argparse
import functools
import os.path

import common.utils
import pyrenderer

from data.necker_ensemble.single_variable import (
    get_file_name_pattern, VARIABLE_NAMES,
    FILE_NAME_TIME_KEY, FILE_NAME_MEMBER_KEY, FILE_NAME_VARIABLE_KEY,
)

from common.utils import parse_range_string, is_valid_index_position


class MultivariateEnsembleDataStorage(object):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('VolumeDataStorage')
        prefix = '--data-storage:'
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
        group.add_argument(prefix + 'variables', type=str, default=None, help=f"""
            Variable keys separated by ':'
            Choices: {VARIABLE_NAMES}
        """)
        group.add_argument(prefix + 'normalization', type=str, default=None, help=f"""
            normalization type to use
            Choices: {VARIABLE_NAMES}
        """)
        group.add_argument(prefix + 'disable-file-verification', action='store_false', dest='verify_files_exist', help="""
        disable file verification for performance reasons
        """)
        group.set_defaults(verify_files_exist=True)

    def __init__(self, timestep_index=None, ensemble_index=None, variables=None, normalization=None, verify_files_exist=True,
                 verbose=False):
        if timestep_index is None:
            timestep_index = '0:1'
        self.timestep_index = parse_range_string(timestep_index)
        if ensemble_index is None:
            ensemble_index = '0:1'
        self.ensemble_index = parse_range_string(ensemble_index)
        if variables is None:
            variables = 'tk'
        self.variable_index = variables.split(':')
        if normalization is None:
            normalization = 'level-min-max'
        self.normalization = normalization
        self.file_pattern = get_file_name_pattern(norm=normalization)
        assert len(set(self.variable_index).difference(set(VARIABLE_NAMES))) ==  0
        if verify_files_exist:
            self._verify_files_exist()

    @classmethod
    def from_dict(cls, args):
        prefix = 'data_storage:'
        return cls(
            timestep_index=args[prefix + 'timestep:index_range'], ensemble_index=args[prefix + 'ensemble:index_range'],
            variables=args[prefix + 'variables'], normalization=args[prefix + 'normalization'],
            verify_files_exist=args['verify_files_exist']
        )

    def _verify_files_exist(self):
        for idx_time in self.timestep_index:
            for idx_ensemble in self.ensemble_index:
                for variable_name in self.variable_index:
                    file_name = self._get_volume_file_name(idx_time, idx_ensemble, variable_name)
                    try:
                        self._load_volume_file(file_name)
                    except Exception() as e:
                        raise Exception(f'[ERROR] Problem while loading volume file {file_name}: {e}')

    def _get_volume_file_name(self, idx_time, idx_ensemble, variable_name):
        query = {
            FILE_NAME_TIME_KEY: idx_time,
            FILE_NAME_MEMBER_KEY: idx_ensemble,
            FILE_NAME_VARIABLE_KEY: variable_name,
        }
        file_name = self.file_pattern.format(**query)
        return file_name

    def num_timesteps(self):
        return len(self.timestep_index)

    def num_members(self):
        return len(self.ensemble_index)

    def num_ensembles(self):
        return self.num_members()

    def num_variables(self):
        return len(self.variable_index)

    @staticmethod
    @functools.lru_cache(4)
    def _load_volume_file(file_name):
        if not os.path.exists(file_name):
            raise ValueError(f'[ERROR] Volume file {file_name} does not exist.')
        vol = pyrenderer.Volume(file_name)
        return vol

    def load_volume(self, timestep=None, ensemble=None, variable=None, index_access=False):
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
        if self.num_variables() > 1:
            assert variable is not None, \
                f'[ERROR] VolumeDataStorage provides {self.num_variables()} variables. Argument <variable> may not be None.'
        if variable is None:
            variable = 0 if index_access else self.variable_index[0]
        if index_access:
            assert is_valid_index_position(timestep,
                                           self.timestep_index), f'[ERROR] Requested time index {timestep} is beyond admissible index range.'
            assert is_valid_index_position(ensemble,
                                           self.ensemble_index), f'[ERROR] Requested ensemble index {ensemble} is beyond admissible index range.'
            assert is_valid_index_position(variable,
                                           self.variable_index), f'[ERROR] Requested variable index {variable} is beyond admissible index range.'
            timestep = self.timestep_index[timestep]
            ensemble = self.ensemble_index[ensemble]
            variable = self.variable_index[variable]
        else:
            assert timestep in self.timestep_index, f'[ERROR] Requested time step {timestep} is not available.'
            assert ensemble in self.ensemble_index, f'[ERROR] Requested ensemble member {ensemble} is not available.'
            assert variable in self.variable_index, f'[ERROR] Requested variable {variable} is not available.'
        file_name = self._get_volume_file_name(timestep, ensemble, variable)
        return self._load_volume_file(file_name)

    def min_time(self):
        return min(self.timestep_index)

    def max_time(self):
        return max(self.timestep_index)


def _test_volume_data_storage():
    vds = MultivariateEnsembleDataStorage(
        timestep_index='0:1', ensemble_index='1:4', variables='tk:u:v:w', normalization='global-min-max'
    )
    volume = vds.load_volume(ensemble=0, variable=0, index_access=True)
    feature = volume.get_feature(0)
    level = feature.get_level(0).to_tensor()
    print('Finished')


if __name__ == '__main__':
    _test_volume_data_storage()