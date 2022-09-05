import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.experiment_loading import load_experiment_data
from data.output import get_output_base_path


root_folder = os.path.join(get_output_base_path(), 'ensemble/multi_core')
baseline_experiment_folder = os.path.join(root_folder, 'num_channels')
retraining_experiment_folder = os.path.join(root_folder, 'retraining')
loss_keys = ['l2']


def get_run_names():
    run_names_baseline = os.listdir(baseline_experiment_folder)
    run_names_retraining = os.listdir(retraining_experiment_folder)
    run_names = set(run_names_baseline).intersection(set(run_names_retraining))
    return sorted(rn for rn in list(run_names) if get_member_count(rn) == 64)


def load_baseline_stats(run_name):
    return load_experiment_data(os.path.join(baseline_experiment_folder, run_name), loss_keys)


def load_retrained_stats(run_name):
    return load_experiment_data(os.path.join(retraining_experiment_folder, run_name), loss_keys)


def get_member_count(run_name):
    code = run_name.split('_')[2]
    min_m, max_m = [int(m) for m in code.split('-')]
    return max_m - min_m


def get_label(run_name):
    r_code, c_code, *_ = run_name.split('_')
    r = r_code.replace('-', ':')
    c = int(c_code.split('-')[0])
    m = get_member_count(run_name)
    return f'R: {r}, C: {c}'


def plot_diagonal(ax):
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    max_lims = np.fmax(xlim, ylim)
    min_lims = np.fmin(xlim, ylim)
    ax.plot([max_lims[0], min_lims[1]], [max_lims[0], min_lims[1]], ls="dotted", c="k")


def get_channel_count(run_name):
    c_code = run_name.split('_')[1]
    return int(c_code.split('-')[0])


def get_linestyle(run_name):
    c_code = run_name.split('_')[1]
    if c_code == '32':
        return 'solid'
    elif c_code == '64':
        return 'dashed'
    raise Exception()


def main():
    fig, axs = plt.subplots(1, 2, dpi=300, figsize=(6, 3))

    run_names = get_run_names()
    colors = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
    baseline_stats = {}
    retrained_stats = {}
    for i, run_name in enumerate(run_names):
        baseline_stats[run_name] = load_baseline_stats(run_name)
        retrained_stats[run_name] = load_retrained_stats(run_name)

    def draw_accuracy_plot(ax):

        def draw_graph(i, run_name):
            run_data = baseline_stats[run_name]
            compression_rate = (352 * 250 * 12 * 64) / run_data['num_params']
            ax.plot(compression_rate, np.sqrt(run_data['l2:min_val']), linestyle=get_linestyle(run_name))

        for i, run_name in enumerate(run_names):
            if get_channel_count(run_name) == 32:
                draw_graph(i, run_name)

        ax.set_prop_cycle(None)

        for i, run_name in enumerate(run_names):
            if get_channel_count(run_name) == 64:
                draw_graph(i, run_name)

        ax.set(xscale='log', yscale='log', xlabel='compression rate', ylabel='RMSE (rescaled)', ylim=(4.e-4, 1.5e-2))
        ax.grid()


    def draw_retraining_plot(ax):
        norm_max = np.log(np.max([
            np.max(baseline_stats[run_name]['num_params'].values) / get_member_count(run_name)
            for run_name in run_names
        ]))
        norm_min = np.log(np.min([
            np.min(baseline_stats[run_name]['num_params'].values) / get_member_count(run_name)
            for run_name in run_names
        ]))

        def draw_graph(i, run_name):
            # sizes = np.exp((np.log(baseline_stats[run_name]['num_params'].values) - norm_min) / (norm_max - norm_min))
            sizes = baseline_stats[run_name]['num_params'].values / np.exp(norm_max) + 0.001
            ax.plot(np.sqrt(baseline_stats[run_name]['l2:min_val'].values),
                    np.sqrt(retrained_stats[run_name]['l2:min_val'].values), label=get_label(run_name),
                    zorder=1, linestyle=get_linestyle(run_name))
            ax.scatter(np.sqrt(baseline_stats[run_name]['l2:min_val'].values),
                       np.sqrt(retrained_stats[run_name]['l2:min_val'].values), s=sizes * 2, zorder=2)

        for i, run_name in enumerate(run_names):
            if get_channel_count(run_name) == 32:
                draw_graph(i, run_name)

        ax.set_prop_cycle(None)

        for i, run_name in enumerate(run_names):
            if get_channel_count(run_name) == 64:
                draw_graph(i, run_name)

        ax.set(xscale='log', yscale='log', xlabel='RMSE (rescaled, original members)', ylabel='RMSE (rescaled, retrained)', xlim=(4.e-4, 1.5e-2), ylim=(4.e-4, 1.5e-2))
        plot_diagonal(ax)
        ax.grid()
        ax.legend(loc='lower right')

    draw_accuracy_plot(axs[0])
    draw_retraining_plot(axs[1])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
