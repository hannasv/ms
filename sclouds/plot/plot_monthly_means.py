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

data = xr.open_dataset('/home/hanna/lagrings/results/stats/monthly_mean/monthly_means.nc')

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = TEXT_HEIGHT_IN - 1)
plt.subplots_adjust(hspace = 0.2, top=0.97, bottom=0.03, left = 0.14, right = 0.97)

for var, ax in zip(VARIABLES, axes):
    if var != 'tcc':
        print('Warning this duplicates the RH in plot for tcc')
        vals   = data[var].values
        f_land = data['land_{}'.format(var)].values
        f_sea  = data['sea_{}'.format(var)].values
        date   = data['date_{}'.format(var)].values

    ax.set_title(LONGNAME[var], fontsize = 14)
    ax.plot(date, vals, label = '{}'.format('test'))
    ax.plot(date, f_land, label = '{}'.format('land'))
    ax.plot(date, f_sea, label = '{}'.format('sea'))

    ax.set_ylabel('{} [{}]'.format(var, UNITS[var]))
ax.legend()
plt.savefig(path_python_figures + 'monthly_means.pdf')
