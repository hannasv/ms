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
                                    file_format, add_ticks)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt

""" For version used in thesis

Best model cloud cover

Think about methods to extract a example where the model fits.

"""


################################### example data for taget and prediction

files = glob.glob('/home/hanna/lagrings/ERA5_monthly/*tcc.nc')
#file = '/home/hanna/miphclac/2004_07/2004_07_tcc.nc'
data = xr.open_dataset(files[0])
target = data.isel(time = 0)
prediction = data.isel(time = 1)

################################### example data for era5
path = '/home/hanna/lagrings/ERA5_tcc/'
example = '2018_08_tcc_era.nc'
example_data = xr.open_dataset(path + example)
era_5_data = example_data.sel(time = '2018-08-01T01')['tcc'].values
##########################################################################

n_rows = 1
n_cols = 3

fig, axes =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=False, sharey=True)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

var = 'tcc'

# plot target
vals    = target[var].values
cntours = axes[0].contourf(vals, vmin=0, vmax=1, levels=levels_contourplot,
                                cmap='Blues_r')
axes[0].set_title('Target'.format(var))
# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

# plot predction
p_vals    = prediction[var].values
cntours = axes[1].contourf(p_vals, vmin=0, vmax=1, levels=levels_contourplot,
                                cmap='Blues_r')
axes[1].set_title('Prediction'.format(var))

# plot predction
p_vals    = prediction[var].values
cntours = axes[2].contourf(era_5_data, vmin=0, vmax=1, levels=levels_contourplot,
                            cmap='Blues_r')
axes[2].set_title('ERA5'.format(var))

# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

#fig.colorbar(cntours,  label = '{} [{}]'.format(var, UNITS[var])) # ax=axes,
#a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
fig.colorbar(cntours, ax = axes, anchor = (1.0, 0.0), label = '{} [{}]'.format(var, UNITS[var])) # ax=axes, orientation="horizontal",

axes[0].set_ylabel('Latitude')

axes[0] = add_ticks(axes[0], x_num_tikz = 5, y_num_tikz = 5)
axes[1] = add_ticks(axes[1], x_num_tikz = 5, y_num_tikz = 5)
axes[2] = add_ticks(axes[2], x_num_tikz = 5, y_num_tikz = 5)

axes[0].set_xlabel('Longitude')
axes[1].set_xlabel('Longitude')
axes[2].set_xlabel('Longitude')


plt.subplots_adjust(left=0.1, bottom=0.25, right=0.80, top=0.9, wspace=0.3, hspace=0.1)
plt.savefig(path_python_figures + 'target_prediction_era5_plot_horizonal.pdf')
