""" Plotting routine used to plot subplots of spatially averages monthly means
and filtered by land sea and both.



To regenerate this monthly means file go to

MS/notebooks/stats and run compute_monthly_means

"""
import numpy as np
import xarray as xr

import glob
import os

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)
from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib)
mat = import_matplotlib()
import matplotlib.pyplot as plt

n_rows = len(VARIABLES)
n_cols = 1

data = xr.open_dataset('/home/hanna/lagrings/results/stats/monthly_mean/monthly_means_updated_3.nc')

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 3)
#plt.subplots_adjust(hspace = 0.2, top=0.97, bottom=0.03, left = 0.14, right = 0.97)

for var, ax in zip(VARIABLES, axes):
    vals   = data[var].values
    f_land = data['land_{}'.format(var)].values
    f_sea  = data['sea_{}'.format(var)].values
    date   = data['date_{}'.format(var)].values

    if var == 'tcc':
        from sclouds.io import Filter
        data   = xr.open_dataset('/home/hanna/MS/sclouds/io/2005_04_tcc.nc')
        f_l = Filter('land').set_data(data = data, variable = var)
        f_s  = Filter('sea').set_data(data = data, variable = var)

        mean_all  = np.nanmean(data[var].values)
        mean_land = f_l.get_mean()
        mean_sea  = f_s.get_mean()

        vals[12]   = mean_all
        f_land[12] = mean_land
        f_sea[12]  = mean_sea

    ax.set_title(LONGNAME[var], fontsize = 14)
    ax.plot(date, vals, label = '{}'.format('no filter'))
    ax.plot(date, f_land, label = '{}'.format('land'))
    ax.plot(date, f_sea, label = '{}'.format('sea'))
    if var != 'tcc':
        lab = '{} [{}]'.format(var, UNITS[var])
    else:
        lab = 'cfc [1]'
    ax.set_ylabel(lab)

plt.legend(ncol = 3, frameon = False, bbox_to_anchor=(0.8, -0.25))
plt.subplots_adjust(wspace = 0.2, hspace = 0.3, top=0.9, bottom=0.1, left = 0.14, right = .95)
plt.savefig(path_python_figures + 'monthly_means.pdf')
