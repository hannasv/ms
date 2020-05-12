import os
import glob

import numpy as np
import xarray as xr
import seaborn as sns

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)

from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt

files = glob.glob('/home/hanna/lagrings/ERA5_monthly/*tcc.nc')
#file = '/home/hanna/miphclac/2004_07/2004_07_tcc.nc'
data = xr.open_dataset(files[0])
target = data.isel(time = 0)
prediction = data.isel(time = 1)

n_rows = 2
n_cols = 1

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_HEIGHT_IN)

var = 'tcc'
#fig.suptitle(LONGNAME[var], fontsize = 14)

# plot target
vals    = target[var].values
cntours = axes[0].contourf(vals, levels=levels_contourplot, cmap='Blues_r')
axes[0].set_title('Target')
# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

# plot predction
p_vals    = prediction[var].values
cntours = axes[1].contourf(p_vals, levels=levels_contourplot, cmap='Blues_r')
axes[1].set_title('Prediction')
# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

print('Warming the colorbar is made based on the last subplot---')
#cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
fig.colorbar(cntours, ax = axes, anchor = (1.0, 0.0), label = '{} [{}]'.format(var, UNITS[var])) # ax=axes, orientation="horizontal",
#a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)

axes[0].set_ylabel('Latitude')
axes[1].set_ylabel('Latitude')
axes[1].set_xlabel('Longitude')

axes[0].set_yticklabels(labels = np.linspace(30, 50, 5))
axes[1].set_yticklabels(labels = np.linspace(30, 50, 5))

axes[1].set_xticklabels(labels = np.linspace(-15, 25, 9), rotation = 45)

plt.subplots_adjust(left=0.1, bottom=0.2, right=0.8, top=0.9, wspace=0.1, hspace=0.3)
plt.savefig(path_python_figures + 'target_prediction_plot_vertical.pdf')
