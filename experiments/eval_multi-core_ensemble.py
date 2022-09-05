import os

from training.in_out import directories as io
from training.in_out.multi_run_experiment import MultiRunExperiment

parser = io.build_parser()
parser.add_argument('--grid-resolution', type=str, required=True)
parser.add_argument('--members', type=str, required=True)
parser.add_argument('--core-channels', type=str, default='32:32:32')

args = vars(parser.parse_args())
io.set_debug_mode(args)

resolution = args['grid_resolution']
core_channels = args['core_channels']


folder = resolution.replace(':', '-') + '_' + core_channels.replace(':','-') + '_' + args['members'].replace(':', '-')

EXPERIMENT_NAME = 'ensemble/multi_core/resampling/' + folder
DATA_FILENAME_PATTERN = 'level-min-max_scaling/tk/member{member:04d}/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'
SCRIPT_PATH = 'training/univariate/run_ensemble_training.py'


PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '16*12*352*250',
    '--world-density-data:batch-size': '6*352*250',
    '--world-density-data:validation-share': 0.25,
    '--world-density-data:sub-batching': 1,
    '--lossmode': 'density',
    '--network:core:layer-sizes': core_channels,
    '--network:core:activation': 'SnakeAlt:1',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': [16, 32, 48, 64, 96, 128, 192, 256],
    '--network:latent-features:volume:grid:resolution': resolution,
    '--network:output:parameterization-method': 'mixed',
    '-l1': 1.,
    '--optimizer:lr': 0.01,
    '--optimizer:hyper-params': '{}',
    '--optimizer:scheduler:mode': 'step-lr',
    '--optimizer:scheduler:gamma': 0.2,
    '--optimizer:scheduler:step-lr:step-size': 20,
    '--optimizer:gradient-clipping:max-norm': 1000.,
    '--epochs': 50,
    '--output:save-frequency': 10,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--dataset-resampling:method': 'importance:grid',
    '--importance-sampler:grid:grid-size': '12:11:8',
    '--importance-sampler:grid:num-samples-per-voxel': 16,
    '--dataset-resampling:frequency': 10,
    '--data-storage:ensemble:index-range': args['members'],
}
FLAGS = ['--network:core:split-members']

if __name__ == '__main__':

    project_base_path = io.get_project_base_path()

    output_directory, log_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    experiment = MultiRunExperiment(
        io.INTERPRETER_PATH, os.path.join(project_base_path, SCRIPT_PATH), project_base_path, log_directory
    )

    print('[INFO] Processing grid-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory},
    }
    experiment.process_parameters(parameters_grid_features, flags=FLAGS, randomize=False)

    print('[INFO] Finished')
