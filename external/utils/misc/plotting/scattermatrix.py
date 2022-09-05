import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, TriAnalyzer


def scatter_matrix(data, figure_kws=None, scatter_kws=None, hist_kws=None, contour_kws=None, tri_min_circle=0.01, kde_bw=None):
    num_samples, num_features = data.shape
    if figure_kws is None:
        figure_kws = {}
    if scatter_kws is None:
        scatter_kws = {}
    if hist_kws is None:
        hist_kws = {}
    if contour_kws is None:
        contour_kws = {}
    if kde_bw is None:
        kde_bw = 'scott'
    range_min = np.min(data, axis=0)
    range_max = np.max(data, axis=0)
    middle = (range_max + range_min) / 2
    span = range_max - range_min
    range_max = middle + 1.1 * span / 2
    range_min = middle - 1.1 * span / 2
    fig, ax = plt.subplots(num_features, num_features, **figure_kws)
    fig.subplots_adjust(hspace=0., wspace=0.)
    for i in range(num_features):
        for j in range(num_features):
            if i == j:
                ax[i, j].hist(data[:, i], **hist_kws)
                ax[i, j].set(xlim=[range_min[i], range_max[i]])
            elif j > i:
                ax[i, j].scatter(data[:, j], data[:, i], **scatter_kws)
                ax[i, j].set(xlim=[range_min[j], range_max[j]], ylim=[range_min[i], range_max[i]])
            else:
                ax[i, j].scatter(data[:, j], data[:, i], **scatter_kws)
                # tri = Triangulation(data[:, j], data[:, i])
                # mask = TriAnalyzer(tri).get_flat_tri_mask(tri_min_circle)
                # tri.set_mask(mask)
                # subspace = np.transpose(data[:, [j, i]])
                # kde = st.gaussian_kde(subspace, bw_method=kde_bw).evaluate(subspace)
                # ax[i, j].tricontour(tri, np.log10(kde), **contour_kws)
                ax[i, j].set(xlim=[range_min[j], range_max[j]], ylim=[range_min[i], range_max[i]])
            if i < num_features - 1:
                ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    return fig, ax


if __name__ == '__main__':
    data = np.random.randn(1000, 16)

    fig, ax = scatter_matrix(data, figure_kws={'figsize': (20, 15), 'gridspec_kw': {'wspace': 0.0, 'hspace': 0.0}}, scatter_kws={'alpha':0.01})

    plt.show()
