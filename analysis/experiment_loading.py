import os

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


def compute_parameter_count(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def get_best_loss(h5_file: h5py.File, loss_key: str):
    losses = h5_file[loss_key][...]
    return np.min(losses)


def summarize_loss_data(losses: np.ndarray):
    min_loss_idx = np.argmin(losses)
    min_loss = losses[min_loss_idx]
    final_loss = losses[-1]
    return min_loss, min_loss_idx, final_loss


def load_experiment_data(experiment_directory, loss_keys):

    results_directory = os.path.join(experiment_directory, 'results')
    hdf5_directory = os.path.join(results_directory, 'hdf5')
    model_directory = os.path.join(results_directory, 'model')

    def load_sample_model_for_run(run_name: str):
        run_directory = os.path.join(model_directory, run_name)
        checkpoint_name = next(iter(f for f in os.listdir(run_directory) if f.endswith('.pth')))
        model = torch.load(os.path.join(run_directory, checkpoint_name), map_location='cpu')
        return model['model']

    def get_model_size(run_name: str):
        model = load_sample_model_for_run(run_name)
        return compute_parameter_count(model)

    hdf5_files = sorted(os.listdir(hdf5_directory))
    data = []
    for current_file_name in hdf5_files:
        try:
            current_file = h5py.File(os.path.join(hdf5_directory, current_file_name), mode='r')
        except Exception as err:
            continue
        else:
            run_name = os.path.splitext(current_file_name)[0]
            loss_summary = {}
            for loss_key in loss_keys:
                min_loss, min_loss_idx, final_loss = summarize_loss_data(current_file[loss_key][...])
                loss_summary.update({
                    f'{loss_key}:min_val': min_loss,
                    f'{loss_key}:min_idx': min_loss_idx,
                    f'{loss_key}:final_val': final_loss,
                })
            file_data = {
                'run_name': run_name,
                'num_params': get_model_size(run_name),
                **current_file.attrs,
                **loss_summary
            }
            data.append(file_data)
            current_file.close()
    data = pd.DataFrame(data)
    return data