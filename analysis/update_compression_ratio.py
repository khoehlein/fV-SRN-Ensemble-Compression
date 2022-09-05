import argparse
import os
from functools import lru_cache

import pandas as pd
import torch

from analysis.deviation_statistics import CompressionStatistics


@lru_cache(maxsize=10)
def load_checkpoint(checkpoint_file):
    return torch.load(checkpoint_file, map_location='cpu')['model']


def update_run_statistics(path):
    data = pd.read_csv(path)
    checkpoints = data['checkpoint']
    compression = [
        CompressionStatistics(load_checkpoint(checkpoint_file)).compression_rate()
        for checkpoint_file in checkpoints
    ]
    data['compression_ratio'] = compression
    data.to_csv(path)
    print('[INFO] Finished')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = vars(parser.parse_args())

    base_path = args['path']
    sub_folders = os.listdir(base_path)

    for folder in sub_folders:
        experiment_base_dir = os.path.join(base_path, folder)
        contents = os.listdir(experiment_base_dir)
        if 'stats' in contents:
            print(f'[INFO] Updating run statistics in {experiment_base_dir}')
            update_run_statistics(os.path.join(experiment_base_dir, 'stats', 'run_statistics.csv'))
        else:
            print(f'[INFO] stats not found in {experiment_base_dir}')


if __name__ == '__main__':
    main()
