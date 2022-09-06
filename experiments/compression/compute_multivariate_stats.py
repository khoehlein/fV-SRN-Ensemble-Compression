import os

import numpy as np

from data.necker_ensemble.single_variable import load_ensemble, load_scales
from training.in_out.directories import get_output_base_path
from experiments.compression.compute_compression_stats import evaluate_sz3, evaluate_tthresh, evaluate_zfp


def export_compressor_stats(singleton, ensemble, compressor_name, norm_name):
    out_directory = os.path.join(get_output_base_path(), 'classical_compressors/multivar_uvw')
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    singleton.to_csv(os.path.join(out_directory, f'compressor_stats_{compressor_name}_{norm_name}_univariate.csv'))
    ensemble.to_csv(os.path.join(out_directory, f'compressor_stats_{compressor_name}_{norm_name}_multivariate.csv'))


def load_variables(variables, norm, member=1, timestep=4):
    all_data = []
    all_scales = []
    for variable_name in variables:
        ensemble = load_ensemble(f'{norm}-min-max', variable_name, min_member=member, max_member=member+1, time=timestep)
        all_data.append(ensemble[0])
        scales = load_scales(f'{norm}-min-max', variable_name)
        all_scales.append(scales)
    all_scales = [np.stack(var_scales, axis=0) for var_scales in zip(*all_scales)]
    return all_data, all_scales


def main():
    for norm in ['local', 'level', 'global']:
        data, scales = load_variables(['u', 'v', 'w'], norm)
        print('[INFO] Data shape:', data[0].shape)
        stats = evaluate_sz3(data, scales)
        export_compressor_stats(*stats, 'sz3', norm)
        stats = evaluate_tthresh(data, scales)
        export_compressor_stats(*stats, 'tthresh', norm)
        stats = evaluate_zfp(data, scales)
        export_compressor_stats(*stats, 'zfp', norm)


if __name__ == '__main__':
    main()
