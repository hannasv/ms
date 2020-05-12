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

n_rows = 1
n_cols = 2

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=False, sharey=True)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

var = 'tcc'
#fig.suptitle(LONGNAME[var], fontsize = 14)

# plot target
vals    = target[var].values
cntours = axes[0].contourf(vals, levels=levels_contourplot, cmap='Blues_r')
axes[0].set_title('Target {}'.format(var))
# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

# plot predction
p_vals    = prediction[var].values
cntours = axes[1].contourf(p_vals, levels=levels_contourplot, cmap='Blues_r')
axes[1].set_title('Prediction {}'.format(var))
# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

fig.colorbar(cntours,  label = '{} [{}]'.format(var, UNITS[var])) # ax=axes,
#a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)

axes[0].set_ylabel('Latitude')

axes[0].set_yticklabels(labels = np.linspace(30, 50, 5))

axes[0].set_xticklabels(labels = np.linspace(-15, 25, 9), rotation = 45) # need to fix this
axes[1].set_xticklabels(labels = np.linspace(-15, 25, 9), rotation = 45) # need to fix this done this for precip timeseriesplot
#a.legend()

axes[0].set_xlabel('Longitude')
axes[1].set_xlabel('Longitude')

plt.subplots_adjust(left=0.1, bottom=0.25, right=0.95, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig(path_python_figures + 'target_prediction_plot_horizonal.pdf')
