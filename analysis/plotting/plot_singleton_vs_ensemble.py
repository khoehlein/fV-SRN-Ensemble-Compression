import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from analysis.plotting.plot_compression_stats import draw_compressor_stats
from analysis.plotting.plot_parameter_interplay import plot_single_member_data
from analysis.plotting.plot_retraining_data import get_member_count, get_channel_count

TRAINING_TYPE = 'resampling'


def load_multi_core_data(num_channels=None):
    if num_channels is None:
        num_channels = 64
    base_folder = f'/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/{TRAINING_TYPE}'
    configurations = sorted(os.listdir(base_folder))
    return {
        c: pd.read_csv(os.path.join(base_folder, c, 'stats', 'run_statistics.csv'))
        for c in configurations if get_member_count(c) == 64 and get_channel_count(c) == num_channels
    }


def get_checkpoint_data(data_reduced):
    data_reduced['checkpoint'] = [os.path.split(c)[-1] for c in data_reduced['checkpoint']]
    checkpoints = np.sort(np.unique(data_reduced['checkpoint']))
    valid = data_reduced['checkpoint'] == checkpoints[-1]
    data_reduced = data_reduced.loc[valid, :]
    return data_reduced


def plot_multi_core_data(ax):
    data = load_multi_core_data(num_channels=64)
    for configuration in data:
        data_reduced = get_checkpoint_data(data[configuration])
        compression_rate = data_reduced['compression_ratio'] #(352 * 250 * 12) * data_reduced['num_members'] / data_reduced['num_parameters']
        loss = data_reduced['rmse_reverted']
        ax[0].plot(compression_rate.loc[compression_rate > 1.], loss.loc[compression_rate > 1.])
        loss = data_reduced['dssim_reverted']
        ax[1].plot(compression_rate[compression_rate > 1.], loss[compression_rate > 1.])
    ax[0].set(xscale='log', yscale='log', title='multi-decoder')
    ax[1].set(xscale='log')
    ax[0].grid()
    ax[1].grid()


def load_multi_grid_data(num_channels=None):
    if num_channels is None:
        num_channels = 64
    base_folder = f'/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_grid/{TRAINING_TYPE}'
    configurations = sorted(os.listdir(base_folder))
    return {
        c: pd.read_csv(os.path.join(base_folder, c, 'stats', 'run_statistics.csv'))
        for c in configurations if get_member_count(c) == 64 and get_channel_count(c) == num_channels
    }


def plot_multi_grid_data(ax):
    data = load_multi_grid_data(num_channels=64)
    for configuration in data:
        data_reduced = get_checkpoint_data(data[configuration])
        compression_rate = data_reduced['compression_ratio'] #(352 * 250 * 12) * data_reduced['num_members'] / data_reduced['num_parameters']
        loss = data_reduced['rmse_reverted']
        ax[0].plot(compression_rate.loc[compression_rate > 1.], loss.loc[compression_rate > 1.])
        loss = data_reduced['dssim_reverted']
        ax[1].plot(compression_rate[compression_rate > 1.], loss[compression_rate > 1.])
    ax[0].set(xscale='log', yscale='log', title='multi-grid')
    ax[1].set(xscale='log')
    ax[0].grid()
    ax[1].grid()


def test():
    fig, axs = plt.subplots(2, 1, sharex='all')
    plot_multi_grid_data(axs)
    plt.tight_layout()
    plt.show()


def add_layout(axs):
    for i, ax in enumerate(axs[0]):
        ax.set(ylim=(4.e-7, 2.e0))
        # if i > 0:
        #     ax.set(yticklabels=[])
    for i, ax in enumerate(axs[1]):
        ax.set(ylim=(-0.1, 1.1), xlabel='compression ratio')
        # if i > 0:
        #     ax.set(yticklabels=[])
    axs[0, 0].set(ylabel='RMSE (originalt)')
    axs[1, 0].set(ylabel='DSSIM (original)')


def main():
    fig, axs = plt.subplots(2, 5, sharex='all', figsize=(10,4), dpi=600)
    plot_multi_core_data(axs[:, 3])
    plot_multi_grid_data(axs[:, 4])
    plot_single_member_data(axs[:, 3])
    plot_single_member_data(axs[:, 4])
    draw_compressor_stats(axs[:, :3], ['level'], 'reverted')
    add_layout(axs)
    plt.tight_layout()
    # plt.savefig('compression_accuracy.pdf')
    plt.show()


if __name__ == '__main__':
    main()