import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from training.in_out.directories import get_output_base_path

RESOLUTION_KEY = 'network:latent_features:volume:grid:resolution'
CHECKPOINT_KEY = 'checkpoint'
MARKER = None


def load_single_member_data():
    path = os.path.join(get_output_base_path(), 'single_member/grid_params/parameter_interplay/stats/run_statistics.csv')
    data = pd.read_csv(path)
    drops = []
    for c in data.columns:
        c_unique = np.unique(data[c])
        if len(c_unique) == 1:
            drops.append(c)

    data = data.drop(columns=drops)
    data[CHECKPOINT_KEY] = [os.path.split(c)[-1] for c in data[CHECKPOINT_KEY]]
    data = data.loc[data[CHECKPOINT_KEY] == 'model_epoch_250.pth', :]
    return data


def plot_single_member_data(ax):
    data = load_single_member_data()
    resolutions = np.unique(data[RESOLUTION_KEY])
    for r in resolutions:
        sel = data.loc[data[RESOLUTION_KEY] == r, :]
        sel = sel.groupby(by='num_parameters').mean()
        compression_ratio = sel['compression_ratio'].values #(352*250*12) / sel.index.values
        order = np.argsort(compression_ratio)
        ax[0].plot(compression_ratio[order], sel['rmse_reverted'].values[order], label=r, marker=MARKER, linestyle='--')
        ax[1].plot(compression_ratio[order], sel['dssim_reverted'].values[order], marker=MARKER, linestyle='--')


def add_formatting(axs):
    axs[0].set(xscale='log', yscale='log')
    axs[1].set(xscale='log', ylim=(0.49, 1.1))
    axs[0].legend()


def main():
    fig, axs = plt.subplots(2, 1)
    plot_single_member_data(axs)
    add_formatting(axs)
    plt.show()


if __name__ == '__main__':
    main()
