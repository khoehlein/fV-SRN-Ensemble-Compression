import argparse
import os

import numpy as np
import pandas as pd
import torch

from analysis.deviation_statistics import DeviationStatistics
from data.datasets.univariate import VolumeDataStorage, WorldSpaceDensityEvaluator
from data.necker_ensemble.single_variable import split_file_name_pattern, load_scales, update_file_pattern_base_path
from inference.volume import RenderTool


def _resolution_from_volume(volume):
    return volume.get_feature(0).get_level(0).to_tensor().shape[1:]


def _samples_to_volume_grid(samples, resolution, to_numpy=True):
    out = torch.reshape(samples, resolution)
    if to_numpy:
        out = out.data.cpu().numpy()
    return out


class EvaluateExperiment(object):

    RUN_NAME_KEY = 'run_name'

    def __init__(self, path: str):
        self.path = path

    @property
    def run_base_directory(self):
        return os.path.join(self.path, 'results', 'model')

    def list_run_names(self):
        return list(sorted(f for f in os.listdir(self.run_base_directory) if f.startswith('run')))

    def list_run_parameters(self):
        args = [
            {
                **self.evaluate_run(run_name).get_parameters(),
                'run_name': run_name,
            }
            for run_name in self.list_run_names()
        ]
        return pd.DataFrame(args)

    def _get_run_directory(self, run_name):
        return os.path.join(self.run_base_directory, run_name)

    def evaluate_run(self, run_name):
        return EvaluateRun(self._get_run_directory(run_name))

    def evaluate_all(self, device=None):
        run_names = self.list_run_names()
        data = []
        for run_name in run_names:
            run = self.evaluate_run(run_name)
            run_stats = run.evaluate_all(device=device)
            run_stats[self.RUN_NAME_KEY] = [run_name] * len(run_stats)
            data.append(run_stats)
        return pd.concat(data, axis=0, ignore_index=True)


class EvaluateRun(object):

    def __init__(self, path: str):
        self.path = path

    def get_parameters(self):
        checkpoints = self.list_checkpoints()
        assert len(checkpoints) > 0
        checkpoint = self.evaluate_checkpoint(checkpoints[0])
        return checkpoint.args

    def list_checkpoints(self):
        return sorted(os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.pth'))

    def _get_checkpoint_file_path(self, checkpoint_name):
        return os.path.join(self.path, checkpoint_name)

    def evaluate_checkpoint(self, checkpoint_name, device=None):
        return EvaluateCheckpoint(self._get_checkpoint_file_path(checkpoint_name), device=device)

    def evaluate_all(self, device=None):
        checkpoints = self.list_checkpoints()
        data = []
        for checkpoint_name in checkpoints:
            checkpoint = self.evaluate_checkpoint(checkpoint_name, device=device)
            checkpoint_stats = checkpoint.compute_stats()
            checkpoint_stats.update({'checkpoint': checkpoint_name})
            data.append(checkpoint_stats)
        return pd.DataFrame(data)


class EvaluateCheckpoint(object):

    def __init__(self, path, device=None):
        self.path = path
        self.network = None
        self.args = None
        self.vds = None
        self.positions, self.resolution, self.member_keys = None, None, None
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.load_checkpoint_data()

    def load_checkpoint_data(self):
        checkpoint = torch.load(self.path, map_location=self.device)
        self.args, self.network = checkpoint['parameters'], checkpoint['model']
        pattern_key = 'data_storage:filename_pattern'
        new_pattern = update_file_pattern_base_path(self.args[pattern_key])
        self.args[pattern_key] = new_pattern
        try:
            v = self.args['verify_files_exist']
        except KeyError:
            v = None
        self.args['verify_files_exist'] = False
        self.vds = VolumeDataStorage.from_dict(self.args)
        self.args['verify_files_exist'] = v
        self.positions, self.resolution, self.member_keys = self._get_network_inputs()
        assert len({'num_parameters', 'num_members', 'norm', 'variable'}.intersection(set(self.args.keys()))) == 0
        norm_name, variable_name = split_file_name_pattern(self.vds.file_pattern)
        self.args.update({
            'num_parameters': sum([p.numel() for p in self.network.parameters()]),
            'num_members': self.vds.num_members(),
            'norm': norm_name,
            'variable': variable_name
        })
        self.volume_evaluator = self._build_volume_evaluator()

    def get_parameters(self):
        return self.args.copy()

    def compute_stats(self):
        ground_truth = self._evaluate_ground_truth()
        predictions = self._evaluate_predictions()
        scales = load_scales(self.args['norm'], self.args['variable'])
        stats = DeviationStatistics(ground_truth, predictions, scales=scales).to_dict()
        return {
            **self.args,
            **stats
        }

    def _get_network_inputs(self):
        volume = self.vds.load_volume(0, 0, index_access=True)
        resolution = _resolution_from_volume(volume)
        positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
        positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
        positions = torch.from_numpy(positions).to(self.device)
        member_keys = list(range(self.vds.num_members()))
        return positions, resolution, member_keys

    def _build_volume_evaluator(self):
        render_tool = RenderTool.from_dict(self.args, self.device)
        volume_evaluator = render_tool.get_volume_evaluator()
        if not volume_evaluator.interpolator.grid_resolution_new_behavior:
            volume_evaluator.interpolator.grid_resolution_new_behavior = True
        return volume_evaluator

    def _evaluate_ground_truth(self):
        all_data = []
        for member_key in self.member_keys:
            volume = self.vds.load_volume(ensemble=member_key, timestep=0, index_access=True)
            self.volume_evaluator.set_source(volume_data=volume, mipmap_level=0, feature=None)
            samples = self.volume_evaluator.evaluate(self.positions)
            all_data.append(_samples_to_volume_grid(samples, self.resolution))
        return np.stack(all_data, axis=0)

    def _evaluate_predictions(self):
        all_data = []
        for member_key in self.member_keys:
            network_evaluator = WorldSpaceDensityEvaluator(self.network, 0, 0, member_key)
            samples = torch.clip(network_evaluator.evaluate(self.positions), min=0., max=1.)
            all_data.append(_samples_to_volume_grid(samples, self.resolution))
        return np.stack(all_data, axis=0)


def compute_experiment_stats(path):
    experiment = EvaluateExperiment(path)
    stats = experiment.evaluate_all(device=torch.device('cuda:0'))
    output_folder = os.path.join(path, 'stats')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_file_name = os.path.join(output_folder, 'run_statistics.csv')
    stats.to_csv(output_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = vars(parser.parse_args())
    path = args['path']
    try:
        print(f'[INFO] Evaluating experiment directory {path}')
        compute_experiment_stats(path)
    except Exception:
        print('[INFO] An error occurred. Trying to evaluate sub-folders.')
        subdirs = sorted(os.listdir(path))
        for d in subdirs:
            new_path = os.path.join(path, d)
            try:
                print(f'[INFO] Processing {new_path}')
                compute_experiment_stats(new_path)
            except Exception:
                print(f'[INFO] Failed with processing {new_path}')


if __name__ == '__main__':
    main()
