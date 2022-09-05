import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.output import get_output_base_path

display_names = {
    'sz3': 'SZ3',
    'tthresh': 'TThresh',
    'zfp': 'ZFP',
}


def load_compressor_data(compressor, norm, se):
    file_name = 'classical_compressors/compressor_stats_{}_{}_{}.csv'.format(
        compressor, norm, se)
    data = pd.read_csv(os.path.join(get_output_base_path(), file_name))
    if se == 'ensemble':
        data = data.loc[data['stacking'] == 0, :]
    data = data.loc[np.logical_and(data['compression_ratio'] <= 2000, data['compression_ratio'] > 0.), :]
    return data


def draw_compressor_stats(ax, norms, mode):
    for i, compressor in enumerate(['sz3', 'tthresh', 'zfp']):
        for j, norm in enumerate(norms):
            file_name = 'classical_compressors/compressor_stats_{}_{}_ensemble.csv'.format(compressor, norm)
            data = pd.read_csv(os.path.join(get_output_base_path(), file_name))
            data = data.loc[data['stacking'] == 0, :]
            data = data.loc[np.logical_and(data['compression_ratio'] <= 2000, data['compression_ratio'] > 1.), :]
            ax[0, i].plot(data['compression_ratio'], data[f'rmse_{mode}'], label=f'ensemble')
            ax[1, i].plot(data['compression_ratio'], data[f'dssim_{mode}'], label=f'ensemble')

    for i, compressor in enumerate(['sz3', 'tthresh', 'zfp']):
        ax[0, i].set_prop_cycle(None)
        ax[1, i].set_prop_cycle(None)
        for j, norm in enumerate(norms):

            file_name = 'classical_compressors/compressor_stats_{}_{}_singleton.csv'.format(
                compressor, norm)
            data = pd.read_csv(os.path.join(get_output_base_path(), file_name))
            data = data.loc[np.logical_and(data['compression_ratio'] <= 2000, data['compression_ratio'] > 1.), :]
            # data = data.loc[data['compression_ratio'] <= 1500, :]
            ax[0, i].plot(data['compression_ratio'], data[f'rmse_{mode}'], label=f'single member', linestyle='--')
            ax[1, i].plot(data['compression_ratio'], data[f'dssim_{mode}'], label=f'single member', linestyle='--')

        ax[0, i].set(xscale='log', yscale='log', title=display_names[compressor])
        ax[0, i].grid()
        ax[1, i].set(xscale='log')
        ax[1, i].grid()


def add_axis_labels(ax, mode):
    mode_label = 'original' if mode == 'reverted' else mode
    ax[0, 0].set(ylabel=f'RMSE ({mode_label})')
    ax[1, 0].set(ylabel=f'Data SSIM ({mode_label})')
    ax[1, 1].set(xlabel='Compression ratio')
    ax[1, 1].legend()


def main():
    mode = 'reverted'
    fig, axs = plt.subplots(2, 3, figsize=(10, 5), sharex='all', sharey='row', dpi=300)
    draw_compressor_stats(axs, ['global', 'level', 'local'], mode)
    add_axis_labels(axs, mode)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
