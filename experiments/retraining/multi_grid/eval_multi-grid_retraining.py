import os

from data.output import get_output_base_path
from training.in_out import directories as io
from training.in_out.multi_run_experiment import MultiRunExperiment

parser = io.build_parser()
parser.add_argument('--grid-resolution', type=str, required=True)
parser.add_argument('--old-members', type=str, required=True)
parser.add_argument('--new-members', type=str, required=True)
parser.add_argument('--core-channels', type=int, required=True)
args = vars(parser.parse_args())
io.set_debug_mode(args)

resolution = args['grid_resolution']
core_channels = args['core_channels']

folder = resolution.replace(':', '-') + f'_{core_channels}_' + args['old_members'].replace(':', '-') + '_fast'

EXPERIMENT_NAME = 'ensemble/multi_grid/retraining/' + folder
DATA_FILENAME_PATTERN = 'level-min-max_scaling/tk/member{member:04d}/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'
SCRIPT_PATH = 'experiments/retraining/multi_grid/run_training.py'

models_folder = os.path.join(get_output_base_path(), 'ensemble/multi_grid/num_channels/', folder, '/results/model')
pth_name = 'model_epoch_50.pth'
run_names = [d for d in sorted(os.listdir(models_folder)) if d.startswith('run')]

PARAMETERS = {
    '--multi-grid-loader:pth-path': [os.path.join(models_folder, run_name, pth_name) for run_name in run_names],
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '16*12*352*250',
    '--world-density-data:batch-size': '6*352*250',
    '--world-density-data:validation-share': 0.25,
    '--world-density-data:sub-batching': 1,
    '--lossmode': 'density',
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
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': 10,
    '--data-storage:ensemble:index-range': args['new_members'],
}


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
    experiment.process_parameters(parameters_grid_features, randomize=False)

    print('[INFO] Finished')
