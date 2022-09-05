import os

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from data.necker_ensemble.single_variable import load_ensemble, load_scales, revert_scaling

variable_names = ['tk', 'qv', 'rh']
dim_names = ['lat', 'lon', 'lev']

fig, ax = plt.subplots(6, len(variable_names), gridspec_kw={'hspace': 0.}, figsize=(8, 4), sharex='col')
nbins = 100
bounds = [[210, 280], [-5, 105], [-0.0005, 0.0065]]
for j, variable_name in enumerate(variable_names):
    variable = load_ensemble('global-min-max', variable_name, time=4, min_member=1, max_member=2)[0]
    scales = load_scales('global-min-max', variable_name)
    variable = revert_scaling(variable[None, ...], scales)[0]
    min_val, max_val = np.min(variable), np.max(variable)
    variable = xr.DataArray(
        variable,
        dims=dim_names,
        coords={
            dim: (dim, np.arange(l))
            for dim, l in zip(dim_names, variable.shape)
        }
    )
    bins = np.linspace(min_val, max_val, nbins+1)
    for i, l in enumerate(range(1, 12, 2)):
        selection = variable.isel(lev=11-l)
        ax[i, j].hist(selection.values.ravel(), bins=bins)
        ax[i, j].set(yticklabels=[], yticks=[])
        # if i < 11:
        #     ax[i, j].set(xticklabels=[], xticks=[])
        if j == 0:
            ax[i, 0].set(ylabel=f'l = {12 - l}')


for i in range(len(ax)):
    ax[i, 0].set(xlim=bounds[0])
    # ax[i, 1].set(xlim=bounds[1])
    # ax[i, 2].set(xlim=bounds[2])

ax[-1, 0].set(xlabel='Temperature [K]')
ax[-1, 1].set(xlabel='Water vapor mixing ratio')
ax[-1, 2].set(xlabel='Relative humidity [%]')
plt.tight_layout()
plt.savefig('data_distribution_by_level.pdf')
plt.show()

print('Finished')