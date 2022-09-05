import os

from training.in_out import directories as io
from training.in_out.multi_run_experiment import MultiRunExperiment

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = f'paper/single_member/grid_params/parameter_interplay'

DATA_FILENAME_PATTERN = []
for normalization in ['level']:
    DATA_FILENAME_PATTERN += ['../{}-min-max_scaling/tk/member{:04d}/t04.cvol'.format(normalization, l) for l in range(2, 5)]
SCRIPT_PATH = 'training/univariate/run_singleton_training.py'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'
PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '16*12*352*250',
    '--world-density-data:batch-size': '6*352*250',
    '--world-density-data:validation-share': 0.2,
    '--world-density-data:sub-batching': 8,
    '--lossmode': 'density',
    '--network:core:layer-sizes': '32:32:32',
    '--network:core:activation': 'SnakeAlt:1',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': [2, 4, 8, 12, 16],
    '--network:latent-features:volume:grid:resolution': ['12:176:125', '6:88:63', '6:44:31','3:22:16'],
    '--network:output:parameterization-method': 'mixed',
    '-l1': 1.,
    '--optimizer:lr': 0.01,
    '--optimizer:hyper-params': '{}',
    '--optimizer:scheduler:mode': 'step-lr',
    '--optimizer:scheduler:gamma': 0.2,
    '--optimizer:scheduler:step-lr:step-size': 100,
    '--optimizer:gradient-clipping:max-norm': 1000.,
    '--epochs': 250,
    '--output:save-frequency': 40,
    '--data-storage:filename-pattern': [os.path.join(io.get_data_base_path(), dfp) for dfp in DATA_FILENAME_PATTERN],
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': 50,
}


if __name__ == '__main__':
    project_base_dir = io.get_project_base_path()
    output_directory, log_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    experiment = MultiRunExperiment(
        io.INTERPRETER_PATH, os.path.join(project_base_dir, SCRIPT_PATH), project_base_dir, log_directory
    )

    print('[INFO] Processing grid-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory},
    }
    experiment.process_parameters(parameters_grid_features)

    print('[INFO] Finished')