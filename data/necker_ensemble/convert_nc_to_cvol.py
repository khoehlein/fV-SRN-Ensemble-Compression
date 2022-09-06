import argparse
import os
import struct
from enum import Enum

import numpy as np
import pyrenderer
import torch
import tqdm
import xarray as xr


MEMBER_FILE_PATTERN = 'member{member:04d}.nc'
MEMBER_DIM_NAME = 'member'
TIME_DIM_NAME = 'time'
VARIABLE_NAMES = ['tk', 'rh', 'qv', 'z', 'dbz', 'qhydro', 'u', 'v', 'w']

BASE_PATH = '/path/to/raw/data'
OUTPUT_PATH = '/path/to/converted/data/{format}/single_variable'


class Axis(Enum):
    TIME = 'time'
    LEVEL = 'lev'
    LATITUDE = 'lat'
    LONGITUDE = 'lon'


DIMENSION_ORDER = [Axis.LONGITUDE.value, Axis.LATITUDE.value, Axis.LEVEL.value]


def int_or_none(value: str):
    try:
        return int(value)
    except ValueError:
        return None


def load_data(variable_name: str, min_member: int, max_member: int, level_slice='-12:'):

    def _load_member(member_id):
        file_name = os.path.join(BASE_PATH, MEMBER_FILE_PATTERN.format(member=member_id))
        dataset = xr.open_dataset(file_name)
        variable = dataset[variable_name].isel({Axis.LEVEL.value: slice(*map(int_or_none, level_slice.split(':')))})
        return variable.expand_dims(dim={MEMBER_DIM_NAME: [member_id]}, axis=0)

    member_ids = list(range(min_member, max_member + 1))
    member_data = []

    with tqdm.tqdm(total=len(member_ids)) as pbar:
        for member_id in member_ids:
            member_data.append(_load_member(member_id))
            pbar.update(1)

    all_data = xr.concat(member_data, dim=MEMBER_DIM_NAME,)

    return all_data


def get_global_min_max_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, TIME_DIM_NAME, Axis.LEVEL.value, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    local_min = all_data.min(dim=summation_dims)
    local_max = all_data.max(dim=summation_dims)
    return local_min, (local_max - local_min)


def get_global_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, TIME_DIM_NAME, Axis.LEVEL.value, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    mu = all_data.mean(dim=summation_dims)
    sigma = all_data.std(dim=summation_dims, ddof=1)
    return mu, sigma


def get_level_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, TIME_DIM_NAME, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    mu = all_data.mean(dim=summation_dims)
    sigma = all_data.std(dim=summation_dims, ddof=1)
    return mu, sigma


def get_level_min_max_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, TIME_DIM_NAME, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    local_min = all_data.min(dim=summation_dims)
    local_max = all_data.max(dim=summation_dims)
    return local_min, (local_max - local_min)


def get_local_min_max_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, TIME_DIM_NAME]
    local_min = all_data.min(dim=summation_dims)
    local_max = all_data.max(dim=summation_dims)
    return local_min, (local_max - local_min)


def convert_variable(variable_name: str, min_member: int, max_member: int, norm='global', format='cvol'):
    data = load_data(variable_name, min_member, max_member)
    if norm == 'global':
        mu, sigma = get_global_normalization(data)
    elif norm == 'level':
        mu, sigma = get_level_normalization(data)
    elif norm == 'local-min-max':
        mu, sigma = get_local_min_max_normalization(data)
    elif norm == 'level-min-max':
        mu, sigma = get_level_min_max_normalization(data)
    elif norm == 'global-min-max':
        mu, sigma = get_global_min_max_normalization(data)
    else:
        raise NotImplementedError()
    sigma = sigma.clip(min=1.e-9)
    normalized_data = (data - mu) / sigma

    variable_dir = os.path.join(OUTPUT_PATH.format(format=format), f'{norm}_scaling', variable_name)

    if not os.path.isdir(variable_dir):
        os.makedirs(variable_dir)

    if Axis.LEVEL.value not in mu.dims:
        mu = mu.expand_dims(Axis.LEVEL.value)
        sigma = sigma.expand_dims(Axis.LEVEL.value)
    if Axis.LATITUDE.value not  in mu.dims:
        mu = mu.expand_dims(Axis.LATITUDE.value)
        sigma = sigma.expand_dims(Axis.LATITUDE.value)
    if Axis.LONGITUDE.value not in mu.dims:
        mu = mu.expand_dims(Axis.LONGITUDE.value)
        sigma = sigma.expand_dims(Axis.LONGITUDE.value)
    mu = mu.transpose(*DIMENSION_ORDER).values
    sigma = sigma.transpose(*DIMENSION_ORDER).values
    np.savez(
        os.path.join(variable_dir, 'scales.npz'),
        offset=mu,
        scale=sigma
    )
    # mu.to_netcdf(os.path.join(scaling_dir, 'offset.nc'))
    # sigma.to_netcdf(os.path.join(scaling_dir, 'scale.nc'))

    for member_id in normalized_data.coords[MEMBER_DIM_NAME]:

        output_path = os.path.join(variable_dir, 'member{:04d}'.format(member_id.values))
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        for timestep_idx, timestep in enumerate(normalized_data.coords[Axis.TIME.value]):
            snapshot = normalized_data.sel({MEMBER_DIM_NAME: member_id.values, Axis.TIME.value: timestep.values})
            snapshot = snapshot.transpose(*DIMENSION_ORDER).values.astype(np.float32)

            if format == 'cvol':
                file_name = 't{:02d}.cvol'.format(timestep_idx)
                vol = pyrenderer.Volume()
                vol.worldX = 10.
                vol.worldY = 10.
                vol.worldZ = 1.
                vol.add_feature_from_tensor(variable_name, torch.from_numpy(snapshot)[None, ...])
                vol.save(os.path.join(output_path, file_name), compression=0)
            elif format == 'dat':
                file_name = 't{:02d}.dat'.format(timestep_idx)
                file_name = os.path.join(output_path, file_name)
                print(f'[INFO] Writing file {file_name}')
                flattened = snapshot.astype(np.float32).ravel().tolist()
                s = struct.pack('d'*len(flattened), *flattened)
                with open(file_name, 'wb') as f:
                    f.write(s)
            else:
                raise NotImplementedError(f'[ERROR] Encountered unknown file format {format}')


    print('Finished')


def convert_ensemble(min_member: int, max_member: int, norm='global', format='cvol'):
    for variable_name in VARIABLE_NAMES:
        print(f'[INFO] Converting variable {variable_name}')
        convert_variable(variable_name, min_member, max_member, norm=norm, format=format)


def convert_all():
    for norm in ['level', 'global', 'local-min-max', 'level-min-max', 'global-min-max']:
        convert_ensemble(1, 128, norm=norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-member', type=int, default=1)
    parser.add_argument('--max-member', type=int, default=128)
    parser.add_argument('--norm', type=str, default='level-min-max',choices=['level', 'global', 'local-min-max', 'level-min-max', 'global-min-max'])
    parser.add_argument('--format', type=str, default='cvol',choices=['cvol', 'dat'])
    args = vars(parser.parse_args())
    print('Format:', args['format'])
    convert_ensemble(args['min_member'], args['max_member'], norm=args['norm'], format=args['format'])


if __name__ == '__main__':
    main()
