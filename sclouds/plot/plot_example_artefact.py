import xarray as xr
import numpy as np
import os


#ll['tcc'].plot()
#plt.show()
""" Plotting routine used to plot subplots of spatially averages monthly means
and filtered by land sea and both.
"""
import numpy as np
import xarray as xr
import seaborn as sns
import glob
import os

from sclouds.helpers import (path_input, path_stats_results, VARIABLES,
                                UNITS, LONGNAME)

from sclouds.plot.helpers import (TEXT_WIDTH_IN, TEXT_HEIGHT_IN,
                                    path_python_figures, import_matplotlib,
                                    cmap_contour_plot, levels_contourplot,
                                    file_format)
mat = import_matplotlib() # for mye
import matplotlib.pyplot as plt

n_rows = 1
n_cols = 1

fig, ax =  plt.subplots(nrows = n_rows, ncols = n_cols, sharex=True, sharey=False)
fig.set_size_inches(w = TEXT_WIDTH_IN, h = 0.5*TEXT_WIDTH_IN)

#file = '/home/hanna/lagrings/ERA5_monthly/2004_07_tcc.nc'
file = '/home/hanna/miphclac/2004_07/2004_07_tcc.nc'
data = xr.open_dataset(file)

subset = data.sel(time = '2004-07-02T12')
var = 'tcc'

vals    = subset[var].values
cntours = ax.contourf(vals, levels=levels_contourplot, cmap='Blues_r')

# Removes white lines
for c in cntours.collections:
    c.set_edgecolor("face")

fig.colorbar(cntours, ax=ax, label = '{} [{}]'.format(var, UNITS[var]))
#a = sns.heatmap(vals, ax = ax, cbar = True, cmap = 'viridis', linewidths=0)
ax.set_title(LONGNAME[var], fontsize = 14)
ax.set_ylabel('Latitude')

ax.set_yticklabels(labels = np.linspace(30, 50, 5))
ax.set_xticklabels(labels = np.linspace(-15, 25, 9), rotation = 45)
#a.legend()
plt.xlabel('Longitude')

plt.subplots_adjust(left=0.1, bottom=0.25, right=0.97, top=0.9, wspace=0.1, hspace=0.1)
plt.savefig(path_python_figures + 'example_artefact.{}'.format(file_format))
