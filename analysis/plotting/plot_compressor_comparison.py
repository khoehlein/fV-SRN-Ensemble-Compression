import matplotlib.pyplot as plt

from analysis.plotting.plot_compression_stats import load_compressor_data
from analysis.plotting.plot_singleton_vs_ensemble import load_multi_grid_data, load_multi_core_data, get_checkpoint_data

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


display_names = {
    'sz3': 'SZ3 (single-member)',
    'zfp': 'ZFP (single-member)',
    'tthresh': 'TThresh (ensemble)'
}


def draw_classical_compressors(ax):
    data = {
        'sz3': load_compressor_data('sz3', 'level', 'singleton'),
        'zfp': load_compressor_data('zfp', 'level', 'singleton'),
        'tthresh': load_compressor_data('tthresh', 'level', 'ensemble'),
    }

    for i, compressor in enumerate(data.keys()):
        cdata = data[compressor]
        ax[0].plot(cdata['compression_ratio'], cdata[f'rmse_reverted'], label=display_names[compressor], color=colors[i], marker='.')
        ax[1].plot(cdata['compression_ratio'], cdata[f'dssim_reverted'], label=display_names[compressor], color=colors[i], marker='.')


def draw_multi_grid_data(ax):
    _draw_model_data(ax, load_multi_grid_data(num_channels=32), colors[3], 'multi-grid')


def draw_multi_core_data(ax):
    _draw_model_data(ax, load_multi_core_data(num_channels=32), colors[4], 'multi-decoder')


def _draw_model_data(ax, data, color, label):
    for i, configuration in enumerate(data.keys()):
        data_reduced = get_checkpoint_data(data[configuration])
        compression_rate = data_reduced['compression_ratio'] #(352 * 250 * 12) * data_reduced['num_members'] / data_reduced['num_parameters']
        label_dict = {'label': label} if i == 0 else {}
        loss = data_reduced['rmse_reverted']
        valid = compression_rate > 0.8
        ax[0].plot(compression_rate.loc[valid], loss.loc[valid], **label_dict, color=color, marker='.')
        loss = data_reduced['dssim_reverted']
        ax[1].plot(compression_rate[valid], loss[valid], **label_dict, color=color, marker='.')


def main():
    fig, axs = plt.subplots(1, 2, sharex='all', figsize=(8,4))
    draw_classical_compressors(axs)
    draw_multi_core_data(axs)
    draw_multi_grid_data(axs)
    axs[0].set(xscale='log', yscale='log', xlabel='compression ratio', title='RMSE (original) $\\downarrow$')
    axs[1].set(xscale='log', xlabel='compression ratio', title='DSSIM (original) $\\uparrow$')
    axs[0].legend()
    axs[0].grid()
    axs[1].grid()
    plt.tight_layout()
    # plt.savefig('all_compressors.pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
